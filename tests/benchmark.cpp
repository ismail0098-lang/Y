// tests/benchmark.cpp
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <cassert>
#include <iomanip>

constexpr size_t CAPACITY = 1024;
constexpr int64_t NUM_OPS = 20000000; // 20 million ops

// 1. Mutex-based queue
template <typename T, size_t Capacity>
class MutexQueue {
    std::queue<T> q;
    std::mutex mtx;
public:
    bool push(const T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (q.size() >= Capacity) return false;
        q.push(item);
        return true;
    }
    bool pop(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (q.empty()) return false;
        item = q.front();
        q.pop();
        return true;
    }
};

// 2. Unaligned lock-free SPSC queue
template <typename T, size_t Capacity>
class UnalignedSpscQueue {
    std::atomic<int64_t> head;
    std::atomic<int64_t> tail;
    T buffer[Capacity];
public:
    UnalignedSpscQueue() : head(0), tail(0) {}
    bool push(const T& item) {
        int64_t current_tail = tail.load(std::memory_order_relaxed);
        int64_t current_head = head.load(std::memory_order_acquire);
        if ((current_tail - current_head) >= Capacity) {
            return false;
        }
        buffer[current_tail & (Capacity - 1)] = item;
        tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& item) {
        int64_t current_head = head.load(std::memory_order_relaxed);
        int64_t current_tail = tail.load(std::memory_order_acquire);
        if (current_head == current_tail) {
            return false;
        }
        item = buffer[current_head & (Capacity - 1)];
        head.store(current_head + 1, std::memory_order_release);
        return true;
    }
};

// 3. Aligned lock-free SPSC queue
template <typename T, size_t Capacity>
class AlignedSpscQueue {
    alignas(64) std::atomic<int64_t> head;
    alignas(64) std::atomic<int64_t> tail;
    T buffer[Capacity];
public:
    AlignedSpscQueue() : head(0), tail(0) {}
    bool push(const T& item) {
        int64_t current_tail = tail.load(std::memory_order_relaxed);
        int64_t current_head = head.load(std::memory_order_acquire);
        if ((current_tail - current_head) >= Capacity) {
            return false;
        }
        buffer[current_tail & (Capacity - 1)] = item;
        tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& item) {
        int64_t current_head = head.load(std::memory_order_relaxed);
        int64_t current_tail = tail.load(std::memory_order_acquire);
        if (current_head == current_tail) {
            return false;
        }
        item = buffer[current_head & (Capacity - 1)];
        head.store(current_head + 1, std::memory_order_release);
        return true;
    }
};

// 3b. SeqCst Aligned SPSC queue (matches Y-compiled memory order)
template <typename T, size_t Capacity>
class SeqCstAlignedSpscQueue {
    alignas(64) std::atomic<int64_t> head;
    alignas(64) std::atomic<int64_t> tail;
    T buffer[Capacity];
public:
    SeqCstAlignedSpscQueue() : head(0), tail(0) {}
    bool push(const T& item) {
        int64_t current_tail = tail.load(std::memory_order_seq_cst);
        int64_t current_head = head.load(std::memory_order_seq_cst);
        if ((current_tail - current_head) >= Capacity) {
            return false;
        }
        buffer[current_tail & (Capacity - 1)] = item;
        tail.store(current_tail + 1, std::memory_order_seq_cst);
        return true;
    }
    bool pop(T& item) {
        int64_t current_head = head.load(std::memory_order_seq_cst);
        int64_t current_tail = tail.load(std::memory_order_seq_cst);
        if (current_head == current_tail) {
            return false;
        }
        item = buffer[current_head & (Capacity - 1)];
        head.store(current_head + 1, std::memory_order_seq_cst);
        return true;
    }
};

// 4. Y-compiled queue interface
struct YSpscBuffer {
    alignas(64) std::atomic<int64_t> head; // index 0
    int64_t _pad1[7];                      // index 1-7
    alignas(64) std::atomic<int64_t> tail; // index 8
    int64_t _pad2[7];                      // index 9-15
    int64_t buffer[1024];                  // index 16 (inline array)
};

extern "C" {
    int32_t spsc_push(YSpscBuffer* s, int64_t item);
    int32_t spsc_pop(YSpscBuffer* s, int64_t* item_ref);
}

template <typename Queue>
void benchmark_queue(const std::string& name, Queue& q) {
    auto start = std::chrono::high_resolution_clock::now();

    std::thread producer([&]() {
        for (int64_t i = 1; i <= NUM_OPS; ++i) {
            while (!q.push(i)) {
                #if defined(__x86_64__) || defined(_M_X64)
                asm volatile("pause");
                #endif
            }
        }
    });

    std::thread consumer([&]() {
        int64_t item = 0;
        for (int64_t i = 1; i <= NUM_OPS; ++i) {
            while (!q.pop(item)) {
                #if defined(__x86_64__) || defined(_M_X64)
                asm volatile("pause");
                #endif
            }
            if (item != i) {
                std::cerr << "Verification failed at " << i << " (got " << item << ")\n";
                exit(1);
            }
        }
    });

    producer.join();
    consumer.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double ops_per_sec = NUM_OPS / diff.count();

    std::cout << std::left << std::setw(30) << name 
              << "Time: " << std::fixed << std::setprecision(3) << diff.count() << " s | "
              << "Throughput: " << std::setprecision(2) << (ops_per_sec / 1e6) << " MOps/s\n";
}

// Wrapper for Y-compiled queue to match template interface
class YQueueWrapper {
    YSpscBuffer y_q;
public:
    YQueueWrapper() {
        y_q.head.store(0, std::memory_order_relaxed);
        y_q.tail.store(0, std::memory_order_relaxed);
    }
    ~YQueueWrapper() {
    }
    bool push(int64_t item) {
        return spsc_push(&y_q, item) == 1;
    }
    bool pop(int64_t& item) {
        return spsc_pop(&y_q, &item) == 1;
    }
};

int main() {
    std::cout << "==================================================\n";
    std::cout << " Lock-Free SPSC Ring Buffer Benchmark\n";
    std::cout << " Operations: " << NUM_OPS << " | Capacity: " << CAPACITY << "\n";
    std::cout << "==================================================\n";

    // 1. Mutex Queue
    {
        MutexQueue<int64_t, CAPACITY> q;
        benchmark_queue("Mutex-based std::queue", q);
    }

    // 2. Unaligned SPSC Queue
    {
        UnalignedSpscQueue<int64_t, CAPACITY> q;
        benchmark_queue("Unaligned C++ SPSC Queue", q);
    }

    // 3. Aligned SPSC Queue
    {
        AlignedSpscQueue<int64_t, CAPACITY> q;
        benchmark_queue("Aligned C++ SPSC Queue", q);
    }

    // 3b. SeqCst Aligned SPSC Queue
    {
        SeqCstAlignedSpscQueue<int64_t, CAPACITY> q;
        benchmark_queue("SeqCst Aligned C++ SPSC Queue", q);
    }

    // 4. Y-compiled Aligned SPSC Queue
    {
        YQueueWrapper q;
        benchmark_queue("Y-compiled Aligned SPSC Queue", q);
    }

    std::cout << "==================================================\n";
    return 0;
}
