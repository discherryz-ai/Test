// DEBUG: wait profiler (see waitprof_patch.py)
#pragma once

#include <array>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <cxxabi.h>
#include <sys/prctl.h>
#include <common.h>

namespace skyline::dbg {
    struct WaitSite {
        std::atomic<const void *> site{};
        const char *tag{};
        std::atomic<u64> ns{};
        std::atomic<u32> count{};
        char thread[17]{};
    };

    constexpr size_t WaitSiteCount{128};

    inline std::array<WaitSite, WaitSiteCount> &WaitSites() {
        static std::array<WaitSite, WaitSiteCount> sites;
        return sites;
    }

    inline u64 WaitNowNs() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    inline void WaitReport(u64 windowNs) {
        struct Row { u64 ns; u32 count; const void *site; const char *tag; char thread[17]; };
        std::array<Row, WaitSiteCount> rows{};
        size_t used{};
        u64 total{};
        for (auto &entry : WaitSites()) {
            const void *site{entry.site.load(std::memory_order_relaxed)};
            if (!site)
                continue;
            u64 ns{entry.ns.exchange(0)};
            u32 count{entry.count.exchange(0)};
            if (!ns)
                continue;
            total += ns;
            Row &row{rows[used++]};
            row = {ns, count, site, entry.tag, {}};
            std::memcpy(row.thread, entry.thread, sizeof(row.thread) - 1);
        }
        std::sort(rows.begin(), rows.begin() + static_cast<long>(used), [](const Row &a, const Row &b) { return a.ns > b.ns; });
        LOGW("DBG waitprof window={:.2f}s total={:.0f}ms sites={}", static_cast<double>(windowNs) / 1e9, static_cast<double>(total) / 1e6, used);
        for (size_t i{}; i < std::min<size_t>(used, 6); i++) {
            const Row &row{rows[i]};
            Dl_info info{};
            std::string name{"?"};
            if (dladdr(row.site, &info) && info.dli_sname) {
                int status{};
                char *demangled{abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status)};
                name = (status == 0 && demangled) ? demangled : info.dli_sname;
                std::free(demangled);
                if (name.size() > 110)
                    name.resize(110);
            }
            LOGW("DBG waitprof   [{}] {} thread={} ms={:.1f} n={}", row.tag ? row.tag : "?", name, row.thread, static_cast<double>(row.ns) / 1e6, row.count);
        }
    }

    inline void WaitRecord(const void *site, const char *tag, u64 ns) {
        static std::atomic<u64> windowStart{};
        if (ns >= 20000) { // Ignore waits shorter than 20us
            auto &sites{WaitSites()};
            size_t hash{(reinterpret_cast<uintptr_t>(site) >> 2) % WaitSiteCount};
            for (size_t i{}; i < WaitSiteCount; i++) {
                auto &entry{sites[(hash + i) % WaitSiteCount]};
                const void *current{entry.site.load(std::memory_order_acquire)};
                if (current == nullptr && entry.site.compare_exchange_strong(current, site))
                    current = site;
                if (current != site)
                    continue;
                if (!entry.tag)
                    entry.tag = tag;
                entry.ns.fetch_add(ns, std::memory_order_relaxed);
                entry.count.fetch_add(1, std::memory_order_relaxed);
                prctl(PR_GET_NAME, entry.thread, 0, 0, 0);
                break;
            }
        }
        u64 now{WaitNowNs()};
        u64 start{windowStart.load(std::memory_order_relaxed)};
        if (!start) {
            windowStart.compare_exchange_strong(start, now);
        } else if (now - start >= 1000000000ULL && windowStart.compare_exchange_strong(start, now)) {
            WaitReport(now - start);
        }
    }

    inline thread_local int waitScopeDepth{};

    /**
     * @brief Measures how long the enclosing scope blocks, only the outermost scope on a thread is recorded
     */
    struct WaitScope {
        const void *site;
        const char *tag;
        u64 start{};
        bool active;

        WaitScope(const void *site, const char *tag) : site{site}, tag{tag}, active{waitScopeDepth++ == 0} {
            if (active)
                start = WaitNowNs();
        }

        ~WaitScope() {
            waitScopeDepth--;
            if (active)
                WaitRecord(site, tag, WaitNowNs() - start);
        }
    };
}
