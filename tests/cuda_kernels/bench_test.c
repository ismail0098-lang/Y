#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
typedef struct {
    int64_t head; int64_t p1,p2,p3,p4,p5,p6,p7;
    int64_t tail; int64_t p8,p9,p10,p11,p12,p13,p14;
    int64_t buffer[1024];
} YQ;
extern int32_t spsc_push(YQ* s, int64_t item);
extern int32_t spsc_pop(YQ* s, int64_t* out);
int main(){
    YQ* q = aligned_alloc(64, sizeof(YQ));
    memset(q,0,sizeof(YQ));
    printf("push(42)=%d\n", spsc_push(q,42));
    int64_t v=0;
    printf("pop()=%d val=%lld\n", spsc_pop(q,&v),(long long)v);
    free(q); return 0;
}
