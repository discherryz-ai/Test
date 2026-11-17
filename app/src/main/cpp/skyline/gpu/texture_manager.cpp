// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <sys/stat.h>
#include <common/trace.h>
#include <gpu.h>
#include "texture_manager.h"
#include <gpu/dbg_toggle.h>

namespace skyline::gpu {
    TextureManager::TextureManager(GPU &gpu) : gpu(gpu) {}

    void TextureManager::DbgDumpAll(const std::string &directory, size_t maxBytes, ContextTag tag) {
        mkdir(directory.c_str(), 0777);

        std::vector<std::shared_ptr<Texture>> unique;
        unique.reserve(textures.size());
        for (const auto &mapping : textures)
            if (std::find(unique.begin(), unique.end(), mapping.texture) == unique.end())
                unique.push_back(mapping.texture);

        auto indexPath{directory + "/index.txt"};
        FILE *index{fopen(indexPath.c_str(), "w")};
        if (!index) {
            LOGE("DBG texdump: can't open {}: {}", indexPath, strerror(errno));
            return;
        }

        constexpr size_t MaxTotalBytes{384ULL * 1024 * 1024};
        size_t dumpedCount{}, dumpedBytes{}, skippedCount{};
        fputs("# id guestAddr guestFormat hostFormat WxHxD levels layers tiling layout dirty scaled status file\n", index);

        for (size_t i{}; i < unique.size(); i++) {
            auto &texture{*unique[i]};
            std::string status{"ok"}, fileName{"-"};

            bool locked{false};
            const bool ownedByCaller{tag && texture.tag.load() == tag};
            if (!ownedByCaller) {
                if (texture.try_lock())
                    locked = true;
                else
                    status = "LOCKED_BY_OTHER";
            }

            if (status == "ok") {
                try {
                    size_t aspectCount{};
                    for (auto aspect : {vk::ImageAspectFlagBits::eColor, vk::ImageAspectFlagBits::eDepth, vk::ImageAspectFlagBits::eStencil})
                        if (texture.format->vkAspect & aspect)
                            aspectCount++;

                    size_t hostSize{};
                    for (const auto &level : texture.mipLayouts)
                        hostSize += level.targetLinearSize * texture.layerCount;
                    hostSize *= aspectCount;

                    if (!texture.GetBacking() || texture.layout == vk::ImageLayout::eUndefined)
                        status = "UNDEFINED_CONTENT";
                    else if (!(texture.format->vkAspect & vk::ImageAspectFlagBits::eColor) || aspectCount != 1)
                        status = "SKIPPED_NOT_COLOUR";
                    else if (hostSize == 0 || hostSize > maxBytes)
                        status = "SKIPPED_SIZE_" + std::to_string(hostSize);
                    else if (dumpedBytes + hostSize > MaxTotalBytes)
                        status = "SKIPPED_TOTAL_LIMIT";

                    if (status == "ok") {
                        auto staging{gpu.memory.AllocateStagingBuffer(hostSize)};
                        texture.WaitOnFence();
                        auto copyCycle{gpu.scheduler.Submit([&](vk::raii::CommandBuffer &commandBuffer) {
                            texture.CopyIntoStagingBuffer(commandBuffer, staging);
                        })};
                        copyCycle->Wait();

                        fileName = fmt::format("{:04}_{:x}_{}x{}_{}.bin", i, texture.guest ? reinterpret_cast<uintptr_t>(texture.guest->mappings.front().data()) : 0, texture.dimensions.width, texture.dimensions.height, vk::to_string(texture.format->vkFormat));
                        auto filePath{directory + "/" + fileName};
                        if (FILE *file{fopen(filePath.c_str(), "wb")}) {
                            fwrite(staging->data(), 1, hostSize, file);
                            fclose(file);
                            dumpedCount++;
                            dumpedBytes += hostSize;
                        } else {
                            status = std::string{"WRITE_FAILED_"} + strerror(errno);
                            fileName = "-";
                        }
                    }
                } catch (const std::exception &e) {
                    status = std::string{"EXCEPTION_"} + e.what();
                }
            }

            if (status != "ok")
                skippedCount++;

            auto line{fmt::format("{} 0x{:x} {} {} {}x{}x{} {} {} {} {} {} {} {} {}\n",
                                  i,
                                  texture.guest ? reinterpret_cast<uintptr_t>(texture.guest->mappings.front().data()) : 0,
                                  texture.guest ? vk::to_string(texture.guest->format->vkFormat) : "-",
                                  vk::to_string(texture.format->vkFormat),
                                  texture.dimensions.width, texture.dimensions.height, texture.dimensions.depth,
                                  texture.levelCount, texture.layerCount,
                                  vk::to_string(texture.tiling), vk::to_string(texture.layout),
                                  static_cast<int>(texture.dirtyState),
                                  texture.scaledRenderTarget ? 1 : 0,
                                  status, fileName)};
            fputs(line.c_str(), index);

            if (locked)
                texture.unlock();
        }

        fclose(index);
        LOGW("DBG texdump: {} textures total, {} dumped ({} KiB), {} skipped -> {}", unique.size(), dumpedCount, dumpedBytes / 1024, skippedCount, directory);
    }

    std::shared_ptr<TextureView> TextureManager::FindOrCreate(const GuestTexture &guestTexture, ContextTag tag) {
        TRACE_EVENT("gpu", "TextureManager::FindOrCreate");
        if (dbg::Mask() & 0x400u)
            dbg::WaitLastSubmit();

        auto guestMapping{guestTexture.mappings.front()};

        /*
         * Iterate over all textures that overlap with the first mapping of the guest texture and compare the mappings:
         * 1) All mappings match up perfectly, we check that the rest of the supplied mappings correspond to mappings in the texture
         * 1.1) If they match as well, we check for format/dimensions/tiling config matching the texture and return or move onto (3)
         * 2) Only a contiguous range of mappings match, we check for if the overlap is meaningful with layout math, it can go two ways:
         * 2.1) If there is a meaningful overlap, we check for format/dimensions/tiling config compatibility and return or move onto (3)
         * 2.2) If there isn't, we move onto (3)
         * 3) If there's another overlap we go back to (1) with it else we go to (4)
         * 4) We check all the overlapping texture for if they're in the texture pool:
         * 4.1) If they are, we do nothing to them
         * 4.2) If they aren't, we delete them from the map
         * 5) Create a new texture and insert it in the map then return it
         */

        std::shared_ptr<Texture> match{};
        boost::container::small_vector<std::shared_ptr<Texture>, 4> matches{};
        auto mappingEnd{std::upper_bound(textures.begin(), textures.end(), guestMapping, [guestMapping](const auto &value, const auto &element) {
            return guestMapping.end() < element.end();
        })}, hostMapping{std::lower_bound(mappingEnd, textures.end(), guestMapping, [guestMapping](const auto &value, const auto &element) {
            return guestMapping.begin() < element.end();
        })};

        std::shared_ptr<Texture> fullMatch{};
        std::shared_ptr<Texture> layerMipMatch{};
        u32 matchLevel{};
        u32 matchLayer{};

        while (hostMapping != textures.begin() && (--hostMapping)->end() > guestMapping.begin()) {
            auto &hostMappings{hostMapping->texture->guest->mappings};
            if (!hostMapping->contains(guestMapping) || hostMapping->texture->replaced)
                continue;

            // We need to check that all corresponding mappings in the candidate texture and the guest texture match up
            // Only the start of the first matched mapping and the end of the last mapping can not match up as this is the case for views
            auto firstHostMapping{hostMapping->iterator};
            auto lastGuestMapping{guestTexture.mappings.back()};
            auto lastHostMapping{std::find_if(firstHostMapping, hostMappings.end(), [&lastGuestMapping](const span<u8> &it) {
                return lastGuestMapping.begin() > it.begin() && lastGuestMapping.end() > it.end();
            })}; //!< A past-the-end iterator for the last host mapping, the final valid mapping is prior to this iterator
            bool mappingMatch{std::equal(firstHostMapping, lastHostMapping, guestTexture.mappings.begin(), guestTexture.mappings.end(), [](const span<u8> &lhs, const span<u8> &rhs) {
                return lhs.end() == rhs.end(); // We check end() here to implicitly ignore any offset from the first mapping
            })};

            if (firstHostMapping == hostMappings.begin() && firstHostMapping->begin() == guestMapping.begin() && mappingMatch && lastHostMapping == hostMappings.end() && lastGuestMapping.end() == std::prev(lastHostMapping)->end()) {
                // We've gotten a perfect 1:1 match for *all* mappings from the start to end, we just need to check for compatibility aside from this
                auto &matchGuestTexture{*hostMapping->texture->guest};
                if (matchGuestTexture.format->IsCompatible(*guestTexture.format) &&
                    ((((matchGuestTexture.dimensions.width == guestTexture.dimensions.width &&
                        matchGuestTexture.dimensions.height == guestTexture.dimensions.height) || matchGuestTexture.CalculateLayerSize() == guestTexture.CalculateLayerSize()) &&
                        matchGuestTexture.GetViewDepth() <= guestTexture.GetViewDepth())
                        || matchGuestTexture.viewMipBase > 0)
                    && matchGuestTexture.tileConfig == guestTexture.tileConfig) {
                    fullMatch = hostMapping->texture;
                } else {
                    matches.push_back(hostMapping->texture);
                }
            } else {
                auto &matchGuestTexture{*hostMapping->texture->guest};
                if (matchGuestTexture.format->IsCompatible(*guestTexture.format) && matchGuestTexture.tileConfig == guestTexture.tileConfig &&
                        (!layerMipMatch || (matchGuestTexture.GetViewLayerCount() >= layerMipMatch->guest->GetViewLayerCount() && matchGuestTexture.mipLevelCount >= layerMipMatch->guest->mipLevelCount))) {
                    size_t memOffset{static_cast<size_t>(guestMapping.data() - hostMapping->texture->guest->mappings.front().data())};
                    size_t layerMemOffset{};
                    bool matched{};
                    for (u32 layer{}; layer < hostMapping->texture->layerCount; layer++) {
                        u32 level{};
                        size_t levelMemOffset{};

                        for (auto &mipLevel : hostMapping->texture->mipLayouts) {
                            if (layerMemOffset + levelMemOffset == memOffset) {
                                if (mipLevel.blockLinearSize == guestTexture.CalculateLayerSize()) {
                                    matched = true;
                                    matchLayer = layer;
                                    matchLevel = level;
                                    break;
                                }
                                level++;
                                levelMemOffset += mipLevel.blockLinearSize;
                            }
                        }

                        if (matched)
                            break;
                        layerMemOffset += matchGuestTexture.GetLayerStride();
                    }

                    if (matched) {
                        if (layerMipMatch)
                            layerMipMatch->replaced = true;

                        if (fullMatch)
                            fullMatch->replaced = true;

                        layerMipMatch = hostMapping->texture;
                    }
                }
            }
         }

        {
            const bool dbgChosen{static_cast<bool>(layerMipMatch) || static_cast<bool>(fullMatch)};
            if (!matches.empty() || !dbgChosen) {
                static u64 dbgKeys[256]{};
                static u32 dbgCounts[256]{};
                static u32 dbgUsed{};
                const u64 chosenPtr{static_cast<u64>(reinterpret_cast<uintptr_t>(layerMipMatch ? layerMipMatch.get() : fullMatch.get()))};
                u64 aPtr{}, aW{}, aH{};
                int aFmt{};
                if (!matches.empty() && matches[0]->guest) {
                    aPtr = static_cast<u64>(reinterpret_cast<uintptr_t>(matches[0].get()));
                    aW = static_cast<u64>(matches[0]->guest->dimensions.width);
                    aH = static_cast<u64>(matches[0]->guest->dimensions.height);
                    aFmt = static_cast<int>(matches[0]->guest->format->vkFormat);
                }
                const u64 parts[]{static_cast<u64>(reinterpret_cast<uintptr_t>(guestMapping.data())), static_cast<u64>(guestMapping.size()), static_cast<u64>(static_cast<int>(guestTexture.format->vkFormat)), static_cast<u64>(guestTexture.dimensions.width), static_cast<u64>(guestTexture.dimensions.height), chosenPtr, static_cast<u64>(matches.size()), aPtr};
                u64 key{1469598103934665603ULL};
                for (u64 part : parts)
                    key = (key ^ part) * 1099511628211ULL;
                u32 idx{dbgUsed};
                for (u32 i{}; i < dbgUsed; i++)
                    if (dbgKeys[i] == key) { idx = i; break; }
                if (idx == dbgUsed && dbgUsed < 256) { dbgKeys[dbgUsed] = key; dbgUsed++; }
                if (idx < 256) {
                    u32 n{++dbgCounts[idx]};
                    if (n <= 5 || (n & (n - 1)) == 0)
                        LOGW("DBG TexMgr {} n={} addr=0x{:x} size={} req={}x{} fmt={} chosen=0x{:x} aliases={} alias0=0x{:x} {}x{} fmt={}", dbgChosen ? "ALIAS_IGNORED" : "CREATE", n, parts[0], parts[1], static_cast<u32>(guestTexture.dimensions.width), static_cast<u32>(guestTexture.dimensions.height), static_cast<int>(guestTexture.format->vkFormat), chosenPtr, matches.size(), aPtr, static_cast<u32>(aW), static_cast<u32>(aH), aFmt);
                }
            }
        }
        if (layerMipMatch) {
            ContextLock textureLock{tag, *layerMipMatch};
            return layerMipMatch->GetView(guestTexture.viewType, vk::ImageSubresourceRange{
                .aspectMask = guestTexture.aspect,
                .baseMipLevel = guestTexture.viewMipBase + matchLevel,
                .levelCount = guestTexture.viewMipCount,
                .baseArrayLayer = guestTexture.baseArrayLayer + matchLayer,
                .layerCount = guestTexture.GetViewLayerCount(),
            }, guestTexture.format, guestTexture.swizzle);
        } else if (fullMatch) {
            ContextLock textureLock{tag, *fullMatch};
            return fullMatch->GetView(guestTexture.viewType, vk::ImageSubresourceRange{
                .aspectMask = guestTexture.aspect,
                .baseMipLevel = guestTexture.viewMipBase,
                .levelCount = guestTexture.viewMipCount,
                .baseArrayLayer = guestTexture.baseArrayLayer,
                .layerCount = guestTexture.GetViewLayerCount(),
            }, guestTexture.format, guestTexture.swizzle);
        }

        for (auto &texture : matches)
            texture->SynchronizeGuest(false, true);

        // Create a texture as we cannot find one that matches
        auto texture{std::make_shared<Texture>(gpu, guestTexture)};
        texture->SetupGuestMappings();
        texture->TransitionLayout(vk::ImageLayout::eGeneral);
        auto it{texture->guest->mappings.begin()};
        textures.emplace(mappingEnd, TextureMapping{texture, it, guestMapping});
        while ((++it) != texture->guest->mappings.end()) {
            guestMapping = *it;
            auto mapping{std::upper_bound(textures.begin(), textures.end(), guestMapping)};
            // TODO: Delete overlapping textures that aren't in texture pool
            textures.emplace(mapping, TextureMapping{texture, it, guestMapping});
        }

        return texture->GetView(guestTexture.viewType, vk::ImageSubresourceRange{
            .aspectMask = guestTexture.aspect,
            .baseMipLevel = guestTexture.viewMipBase,
            .levelCount = guestTexture.viewMipCount,
            .baseArrayLayer = guestTexture.baseArrayLayer,
            .layerCount = guestTexture.GetViewLayerCount(),
        }, guestTexture.format, guestTexture.swizzle);
    }
}
