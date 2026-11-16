// SPDX-License-Identifier: MPL-2.0
// Диагностика: переключатели, читаемые во время работы (системное свойство или файл). В обычной работе все равны 0.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <functional>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
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
     * @note ПО УМОЛЧАНИЮ после каждой отправки команд ждём, пока поток записи запишет и отправит буфер команд
     * @note Бит 0x800000: колбэки/семафоры/синхроточки выполняет отдельный поток после отправки буфера команд (поток GPFIFO не блокируется)
     * @note Бит 0x4000: выключить это ожидание; 0x200 / 0x400 / 0x1000: ждать только при поиске буферов / текстур / перед 2D-блитом
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

    inline std::mutex &LastSubmitMutex() {
        static std::mutex mutex;
        return mutex;
    }

    inline std::function<void()> &LastSubmitWaiter() {
        static std::function<void()> waiter;
        return waiter;
    }

    /**
     * @brief Запоминает, как дождаться записи и отправки последнего буфера команд
     */
    inline void SetLastSubmit(std::function<void()> waiter) {
        std::scoped_lock lock{LastSubmitMutex()};
        LastSubmitWaiter() = std::move(waiter);
    }

    /**
     * @brief Ждёт, пока последний отправленный буфер команд будет записан и отправлен в очередь GPU
     */
    inline void WaitLastSubmit() {
        std::function<void()> waiter;
        {
            std::scoped_lock lock{LastSubmitMutex()};
            waiter = LastSubmitWaiter();
        }
        if (waiter) {
            static std::atomic<std::uint64_t> calls{0}, blockedUs{0}, blockedCalls{0};
            auto start{std::chrono::steady_clock::now()};
            waiter();
            auto us{static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count())};
            if (us > 50)
                blockedCalls.fetch_add(1, std::memory_order_relaxed);
            blockedUs.fetch_add(us, std::memory_order_relaxed);
            std::uint64_t n{calls.fetch_add(1, std::memory_order_relaxed) + 1};
            if ((n % 2048) == 0)
                LOGW("DBG lazy-wait stats: calls={} blocked_calls={} total_blocked_ms={}", n, blockedCalls.load(std::memory_order_relaxed), blockedUs.load(std::memory_order_relaxed) / 1000);
        }
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

    inline std::atomic<u32> &ExecutorCount() {
        static std::atomic<u32> count{0};
        return count;
    }

    /**
     * @brief Фиксирует создание исполнителя команд (по одному на канал GPU)
     */
    inline void NoteExecutorCreated(const void *owner) {
        u32 n{ExecutorCount().fetch_add(1, std::memory_order_relaxed) + 1};
        LOGW("DBG executor created #{} owner=0x{:x}", n, reinterpret_cast<std::uintptr_t>(owner));
    }

    struct SignalEntry {
        const void *owner;
        std::function<void()> wait; //!< Ждёт, пока последний отправленный этим исполнителем буфер команд попадёт в очередь GPU
    };

    inline std::mutex &SignalMutex() {
        static std::mutex mutex;
        return mutex;
    }

    inline std::vector<SignalEntry> &SignalEntries() {
        static std::vector<SignalEntry> entries;
        return entries;
    }

    /**
     * @brief Исполнитель `owner` сейчас сообщит гостю о завершении работы: запоминаем, чего должны дождаться остальные исполнители перед своей отправкой
     */
    inline void NoteSignal(const void *owner, std::function<void()> wait) {
        std::scoped_lock lock{SignalMutex()};
        for (auto &entry : SignalEntries()) {
            if (entry.owner == owner) {
                entry.wait = std::move(wait);
                return;
            }
        }
        SignalEntries().push_back(SignalEntry{owner, std::move(wait)});
    }

    /**
     * @brief Ждёт, пока все ДРУГИЕ исполнители отправят в очередь GPU буферы команд, о завершении которых они уже сообщили гостю
     */
    inline void WaitOthers(const void *self) {
        std::vector<std::function<void()>> waiters;
        {
            std::scoped_lock lock{SignalMutex()};
            for (auto &entry : SignalEntries())
                if (entry.owner != self && entry.wait)
                    waiters.push_back(entry.wait);
        }
        if (waiters.empty())
            return;

        static std::atomic<std::uint64_t> calls{0}, blocked{0}, blockedUs{0};
        auto start{std::chrono::steady_clock::now()};
        for (auto &waiter : waiters)
            waiter();
        auto us{static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count())};

        std::uint64_t n{calls.fetch_add(1, std::memory_order_relaxed) + 1};
        if (us > 50) {
            blocked.fetch_add(1, std::memory_order_relaxed);
            blockedUs.fetch_add(us, std::memory_order_relaxed);
        }
        if (n <= 5 || (n % 512) == 0)
            LOGW("DBG xexec: calls={} blocked={} blocked_ms={} others={}", n, blocked.load(std::memory_order_relaxed), blockedUs.load(std::memory_order_relaxed) / 1000, waiters.size());
    }

    /**
     * @brief Собирает «как дождаться» для всех ДРУГИХ исполнителей, не ожидая сейчас (ожидание выполнит поток записи через RunWaiters)
     */
    inline std::vector<std::function<void()>> CollectOthers(const void *self) {
        std::vector<std::function<void()>> waiters;
        std::scoped_lock lock{SignalMutex()};
        for (auto &entry : SignalEntries())
            if (entry.owner != self && entry.wait)
                waiters.push_back(entry.wait);
        return waiters;
    }

    /**
     * @brief Выполняет собранные ожидания (вызывается потоком записи перед отправкой буфера команд в очередь GPU) и очищает список
     */
    inline void RunWaiters(std::vector<std::function<void()>> &waiters) {
        if (waiters.empty())
            return;

        static std::atomic<std::uint64_t> calls{0}, blocked{0}, blockedUs{0};
        auto start{std::chrono::steady_clock::now()};
        for (auto &waiter : waiters)
            waiter();
        auto us{static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count())};
        const auto others{waiters.size()};
        waiters.clear();

        std::uint64_t n{calls.fetch_add(1, std::memory_order_relaxed) + 1};
        if (us > 50) {
            blocked.fetch_add(1, std::memory_order_relaxed);
            blockedUs.fetch_add(us, std::memory_order_relaxed);
        }
        if (n <= 5 || (n % 512) == 0)
            LOGW("DBG xexec-rec: calls={} blocked={} blocked_ms={} others={}", n, blocked.load(std::memory_order_relaxed), blockedUs.load(std::memory_order_relaxed) / 1000, others);
    }
}
