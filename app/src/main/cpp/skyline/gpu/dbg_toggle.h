// SPDX-License-Identifier: MPL-2.0
// Диагностика: переключатели, читаемые во время работы (системное свойство или файл). В обычной работе все равны 0.

#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/system_properties.h>
#include <common.h>
#include <logger/logger.h>

namespace skyline::dbg {
    /**
     * @brief Путь к файлу-запасному источнику маски: <внешняя папка приложения>/files/dbg_mask
     */
    inline std::string MaskFilePath() {
        char name[256]{};
        if (FILE *file{std::fopen("/proc/self/cmdline", "r")}) {
            size_t got{std::fread(name, 1, sizeof(name) - 1, file)};
            (void)got;
            std::fclose(file);
        }
        if (char *colon{std::strchr(name, ':')})
            *colon = '\0';
        return std::string{"/storage/emulated/0/Android/data/"} + name + "/files/dbg_mask";
    }

    /**
     * @brief Маска переключателей. Источник: свойство debug.strato.dbg, иначе файл dbg_mask.
     * @note Бит 1: ждать завершения GPU после каждой отправки команд
     * @note Бит 2: не вшивать очистку в loadOp, если в этом подпроходе уже есть отрисовка
     * @note Бит 4: не склеивать подпроходы (каждая отрисовка в своём проходе)
     * @note Бит 16: полный барьер вокруг каждого прохода и каждой команды вне прохода
     * @note Бит 32: выключить мегабуфер
     * @note Бит 64: ждать отправки буфера команд (не выполнения)
     * @note Бит 128: запись в буфер только через GPU-копирование
     * @note Бит 256: перед прямой записью в буфер ждать GPU
     */
    inline u32 Mask() {
        static std::atomic<u32> mask{0};
        static std::atomic<u32> tick{0};
        static std::atomic<u32> lastLogged{0xFFFFFFFFu};
        static const std::string path{MaskFilePath()};

        if ((tick.fetch_add(1, std::memory_order_relaxed) & 0x3Fu) == 0) {
            u32 newMask{};
            const char *source{"none"};

            char value[PROP_VALUE_MAX]{};
            if (__system_property_get("debug.strato.dbg", value) > 0) {
                newMask = static_cast<u32>(std::strtoul(value, nullptr, 0));
                source = "prop";
            } else if (FILE *file{std::fopen(path.c_str(), "r")}) {
                long fromFile{};
                if (std::fscanf(file, "%li", &fromFile) == 1) {
                    newMask = static_cast<u32>(fromFile);
                    source = "file";
                }
                std::fclose(file);
            }

            mask.store(newMask, std::memory_order_relaxed);
            if (lastLogged.exchange(newMask) != newMask)
                LOGW("DBG toggle mask=0x{:x} source={} file={}", newMask, source, path);
        }

        return mask.load(std::memory_order_relaxed);
    }

    /**
     * @brief Фиксирует очистку, пришедшую в подпроход, где уже есть отрисовка (старое поведение: очистка «задним числом» в loadOp, предыдущая отрисовка не стирается)
     */
    inline void NoteClearAfterDraw(bool mergedIntoLoadOp, const char *kind) {
        static std::atomic<u32> total{0};
        static std::atomic<u32> merged{0};
        u32 n{total.fetch_add(1, std::memory_order_relaxed) + 1};
        if (mergedIntoLoadOp)
            merged.fetch_add(1, std::memory_order_relaxed);
        if (n <= 10 || (n & (n - 1)) == 0)
            LOGW("DBG clear-after-draw n={} merged_into_loadop_total={} kind={} merged_now={}", n, merged.load(std::memory_order_relaxed), kind, mergedIntoLoadOp);
    }
}
