#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#pragma pack(push, 1)

// File layout headers matching schema.md exactly
typedef struct {
    char magic[4];
    uint16_t version;
    uint16_t reserved;
    uint32_t char_count;
    uint32_t index_offset;
} KjvgHeader;

typedef struct {
    uint32_t codepoint;
    uint16_t stroke_count;
    uint16_t reserved;
    uint64_t data_offset;
} KjvgIndexEntry;

typedef struct {
    uint16_t x;
    uint16_t y;
} KjvgPoint;

typedef struct {
    uint16_t point_count;
    uint16_t reserved;
    KjvgPoint points[1]; // Flexible array member placeholder
} KjvgStroke;

#pragma pack(pop)

/**
 * Searches the sorted index table of a mapped .kjvg file for the given codepoint.
 * Complexity: O(log C) binary search.
 * Returns pointer to the index entry, or NULL if not found.
 */
const KjvgIndexEntry* kjvg_lookup(const uint8_t* file_ptr, uint32_t codepoint) {
    const KjvgHeader* header = (const KjvgHeader*)file_ptr;
    
    // Safety check magic header
    if (strncmp(header->magic, "KJVG", 4) != 0) {
        return NULL;
    }
    
    const KjvgIndexEntry* index_table = (const KjvgIndexEntry*)(file_ptr + header->index_offset);
    
    int32_t low = 0;
    int32_t high = (int32_t)header->char_count - 1;
    
    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        const KjvgIndexEntry* entry = &index_table[mid];
        
        if (entry->codepoint == codepoint) {
            return entry; // Match found
        } else if (entry->codepoint < codepoint) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    
    return NULL; // Not found
}

/**
 * Traverses the contiguous stroke data segment to retrieve the pointer to a specific stroke.
 * Since stroke lengths are variable, we offset dynamically based on point counts.
 */
const KjvgStroke* kjvg_get_stroke(const uint8_t* file_ptr, const KjvgIndexEntry* entry, uint16_t stroke_idx) {
    if (!entry || stroke_idx >= entry->stroke_count) {
        return NULL;
    }
    
    // Address of first stroke block
    const uint8_t* current_ptr = file_ptr + entry->data_offset;
    
    for (uint16_t i = 0; i < stroke_idx; i++) {
        const KjvgStroke* stroke = (const KjvgStroke*)current_ptr;
        // Offset is: 4 bytes for (point_count + reserved) + (point_count * size_of(KjvgPoint))
        size_t block_size = 4 + (stroke->point_count * sizeof(KjvgPoint));
        current_ptr += block_size;
    }
    
    return (const KjvgStroke*)current_ptr;
}

// Simple test main function demonstrating loading and lookup
int main() {
    FILE* f = fopen("kanji_db.kjvg", "rb");
    if (!f) {
        printf("Please run packer.py first to generate kanji_db.kjvg\n");
        return 1;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    uint8_t* buffer = (uint8_t*)malloc(size);
    if (fread(buffer, 1, size, f) != size) {
        printf("Error reading file.\n");
        fclose(f);
        free(buffer);
        return 1;
    }
    fclose(f);
    
    // Look up character '日' (Unicode codepoint U+065E5)
    uint32_t target_kanji = 0x065E5;
    printf("[Loader] Looking up Kanji 0x%05X...\n", target_kanji);
    
    const KjvgIndexEntry* entry = kjvg_lookup(buffer, target_kanji);
    if (entry) {
        printf("[OK] Found codepoint 0x%05X!\n", entry->codepoint);
        printf("     Strokes count: %d\n", entry->stroke_count);
        printf("     Data offset: %lu\n", (unsigned long)entry->data_offset);
        
        // Print coordinates for each stroke
        for (uint16_t s = 0; s < entry->stroke_count; s++) {
            const KjvgStroke* stroke = kjvg_get_stroke(buffer, entry, s);
            if (stroke) {
                printf("     Stroke #%d: %d points\n", s + 1, stroke->point_count);
                // Print first and last point to verify
                if (stroke->point_count > 0) {
                    // Coordinates normalized to [0, 65535]
                    printf("       Start point: (%u, %u) -> Scaled: (%.2f, %.2f)\n",
                           stroke->points[0].x, stroke->points[0].y,
                           (stroke->points[0].x / 65535.0) * 512.0,
                           (stroke->points[0].y / 65535.0) * 512.0);
                    printf("       End point:   (%u, %u) -> Scaled: (%.2f, %.2f)\n",
                           stroke->points[stroke->point_count - 1].x,
                           stroke->points[stroke->point_count - 1].y,
                           (stroke->points[stroke->point_count - 1].x / 65535.0) * 512.0,
                           (stroke->points[stroke->point_count - 1].y / 65535.0) * 512.0);
                }
            }
        }
    } else {
        printf("[Error] Kanji 0x%05X not found in database.\n", target_kanji);
    }
    
    free(buffer);
    return 0;
}
