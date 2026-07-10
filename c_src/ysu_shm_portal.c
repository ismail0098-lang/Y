#include "ysu_shm_portal.h"

// Implementation of the safe producer for the YSU Engine
void ysu_produce_safe(RingBuffer *rb, float sample) {
    int next_head = (rb->head + 1) % rb->size;

    // Only write if the buffer isn't full
    if (next_head != rb->tail) {
        rb->buffer[rb->head] = sample;
        rb->head = next_head;
    }
}
