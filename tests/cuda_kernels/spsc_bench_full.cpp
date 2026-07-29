#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <stdlib.h>

constexpr size_t CAP  = 1024;
constexpr int64_t OPS = 20'000'000;

// ── 1. Mutex queue ─────────────────────────────────────────
class MutexQueue {
    std::queue<int64_t> q; std::mutex m;
public:
    bool push(int64_t v){std::lock_guard<std::mutex> l(m);if(q.size()>=CAP)return false;q.push(v);return true;}
    bool pop(int64_t& v){std::lock_guard<std::mutex> l(m);if(q.empty())return false;v=q.front();q.pop();return true;}
};

// ── 2. Unaligned C++ SPSC ──────────────────────────────────
class UnalignedSpsc {
    std::atomic<int64_t> head{0}, tail{0}; int64_t buf[CAP];
public:
    bool push(int64_t v){auto t=tail.load(std::memory_order_relaxed),h=head.load(std::memory_order_acquire);if(t-h>=(int64_t)CAP)return false;buf[t&(CAP-1)]=v;tail.store(t+1,std::memory_order_release);return true;}
    bool pop(int64_t& v){auto h=head.load(std::memory_order_relaxed),t=tail.load(std::memory_order_acquire);if(h==t)return false;v=buf[h&(CAP-1)];head.store(h+1,std::memory_order_release);return true;}
};

// ── 3. Aligned C++ SPSC (cache-line separated) ─────────────
class AlignedSpsc {
    alignas(64) std::atomic<int64_t> head{0};
    alignas(64) std::atomic<int64_t> tail{0};
    int64_t buf[CAP];
public:
    bool push(int64_t v){auto t=tail.load(std::memory_order_relaxed),h=head.load(std::memory_order_acquire);if(t-h>=(int64_t)CAP)return false;buf[t&(CAP-1)]=v;tail.store(t+1,std::memory_order_release);return true;}
    bool pop(int64_t& v){auto h=head.load(std::memory_order_relaxed),t=tail.load(std::memory_order_acquire);if(h==t)return false;v=buf[h&(CAP-1)];head.store(h+1,std::memory_order_release);return true;}
};

// ── 4. Y-compiled SPSC (ring_buffer.ysu → LLVM IR → native) ─
// Exact layout from generated IR:
//   %SpscBuffer = type { i64×16, [1024 x i64] }
//   head = field 0 (offset 0),  align 64, atomic acq/rel
//   tail = field 8 (offset 64), align 64, atomic acq/rel
//   buf  = field 16 (offset 128)
typedef struct {
    int64_t head;
    int64_t p1,p2,p3,p4,p5,p6,p7;
    int64_t tail;
    int64_t p8,p9,p10,p11,p12,p13,p14;
    int64_t buffer[1024];
} YSpscBuffer;

extern "C" int32_t spsc_push(YSpscBuffer* s, int64_t item);
extern "C" int32_t spsc_pop (YSpscBuffer* s, int64_t* item_ref);

class YQueue {
    YSpscBuffer* q;
public:
    YQueue(){ q=(YSpscBuffer*)aligned_alloc(64,sizeof(YSpscBuffer)); memset(q,0,sizeof(YSpscBuffer)); }
    ~YQueue(){ free(q); }
    bool push(int64_t v){ return spsc_push(q,v)==1; }
    bool pop(int64_t& v){ return spsc_pop(q,&v)==1; }
};

// ── Runner ──────────────────────────────────────────────────
template<typename Q>
double bench(const char* name, Q& q){
    auto t0 = std::chrono::high_resolution_clock::now();
    std::thread prod([&]{for(int64_t i=1;i<=OPS;i++)while(!q.push(i))asm volatile("pause");});
    std::thread cons([&]{int64_t v=0;for(int64_t i=1;i<=OPS;i++){while(!q.pop(v))asm volatile("pause");if(v!=i){std::cerr<<"VERIFY FAIL i="<<i<<" got "<<v<<"\n";exit(1);}}});
    prod.join(); cons.join();
    double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    double mops = OPS/dt/1e6;
    std::cout << "  " << std::left << std::setw(44) << name
              << std::right << std::setw(7) << std::fixed << std::setprecision(3) << dt << " s"
              << "   " << std::setw(8) << std::setprecision(2) << mops << " MOps/s\n";
    return mops;
}

int main(){
    std::cout << "\n";
    std::cout << "┌──────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│   SPSC Ring Buffer Benchmark — 20 million ops, capacity = 1024   │\n";
    std::cout << "│        Y-compiled ring_buffer.ysu  vs  C++ implementations       │\n";
    std::cout << "│   Hardware: AVX-512 · L2 line 64B · RTX 4070 Ti SUPER           │\n";
    std::cout << "└──────────────────────────────────────────────────────────────────┘\n\n";
    std::cout << "  " << std::left << std::setw(44) << "Implementation"
              << std::right << std::setw(8) << "Time" << "   " << std::setw(12) << "Throughput\n";
    std::cout << "  " << std::string(66, '-') << "\n";

    double m, c1, c2, y;
    { MutexQueue    q; m  = bench("1. Mutex std::queue       [baseline]", q); }
    { UnalignedSpsc q; c1 = bench("2. C++ SPSC  unaligned    [acq/rel]",  q); }
    { AlignedSpsc   q; c2 = bench("3. C++ SPSC  aligned CL64 [acq/rel]",  q); }
    { YQueue        q; y  = bench("4. Y-compiled SPSC        [ring_buffer.ysu]", q); }

    std::cout << "  " << std::string(66, '-') << "\n\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Y vs Mutex baseline:      " << y/m  << "x faster\n";
    std::cout << "  Y vs Unaligned C++ SPSC:  " << y/c1 << "x\n";
    std::cout << "  Y vs Aligned C++ SPSC:    " << y/c2 << "x\n";
    std::cout << "\n  (Y emits atomic acq/rel load/store at align=64 per .ysu source)\n\n";
    return 0;
}
