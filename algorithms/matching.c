#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    float x;
    float y;
} Point2D;

/**
 * Highly optimized, hardware-friendly path-matching algorithm.
 * Computes both spatial point alignment and directional trajectory similarity.
 * Runs in strict O(N) time with O(1) auxiliary space (zero heap allocations).
 * 
 * @param user_path Array of 2D coordinates representing the user's resampled stroke.
 * @param ref_path Array of 2D coordinates representing the reference stroke.
 * @param N Number of points in both paths (resampled to a fixed size, e.g. 32 or 64).
 * @param tolerance_dist Maximum spatial distance tolerance (e.g. 128.0 in 512x512 space).
 * @return Normalized matching score in [0.0, 1.0].
 */
float match_stroke(const Point2D* user_path, const Point2D* ref_path, int32_t N, float tolerance_dist) {
    if (N <= 1) {
        return 0.0f;
    }

    float total_dist = 0.0f;
    float total_dir_sim = 0.0f;
    int32_t valid_dir_segments = 0;

    // Hot loop: Process spatial distance and direction simultaneously
    for (int32_t i = 0; i < N; i++) {
        // 1. Spatial distance (Euclidean distance between corresponding resampled points)
        float dx = user_path[i].x - ref_path[i].x;
        float dy = user_path[i].y - ref_path[i].y;
        total_dist += sqrtf(dx * dx + dy * dy);

        // 2. Directional matching (compare segment tangents)
        if (i < N - 1) {
            float ux = user_path[i+1].x - user_path[i].x;
            float uy = user_path[i+1].y - user_path[i].y;
            float rx = ref_path[i+1].x - ref_path[i].x;
            float ry = ref_path[i+1].y - ref_path[i].y;

            float u_len = sqrtf(ux * ux + uy * uy);
            float r_len = sqrtf(rx * rx + ry * ry);

            // Avoid division by zero for stationary points
            if (u_len > 0.0001f && r_len > 0.0001f) {
                // Dot product of normalized direction vectors
                float dot = (ux * rx + uy * ry) / (u_len * r_len);
                // Clamp dot product to [-1.0, 1.0] due to float precision drift
                if (dot > 1.0f) dot = 1.0f;
                if (dot < -1.0f) dot = -1.0f;

                // Map cosine similarity from [-1.0, 1.0] to [0.0, 1.0]
                float dir_sim = (1.0f + dot) * 0.5f;
                total_dir_sim += dir_sim;
                valid_dir_segments++;
            }
        }
    }

    // Normalized spatial score
    float avg_dist = total_dist / (float)N;
    float spatial_score = 1.0f - (avg_dist / tolerance_dist);
    if (spatial_score < 0.0f) {
        spatial_score = 0.0f;
    }

    // Normalized direction score
    float direction_score = 1.0f;
    if (valid_dir_segments > 0) {
        direction_score = total_dir_sim / (float)valid_dir_segments;
    }

    // Weighted average: 50% spatial layout, 50% stroke drawing direction
    return (0.5f * spatial_score) + (0.5f * direction_score);
}
