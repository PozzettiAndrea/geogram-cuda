// ============================================================
// CUDA kernels for Centroidal Voronoi Tessellation
//
// GPU-accelerated inner loops for Lloyd relaxation and Newton
// optimization of CVT on triangle meshes.
//
// Architecture:
//   - Mesh triangles uploaded once at context creation
//   - Seeds uploaded each iteration
//   - Per-triangle kernels: assign to nearest seed, accumulate
//   - Per-seed reduction via atomicAdd (double precision)
//   - Point update kernel normalizes mg/m
//
// Target: SM 86+ (RTX A6000), double-precision atomics
// ============================================================

#include "cvt_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>

// ============================================================
// Error checking
// ============================================================

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "[CVT_CUDA] %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        return;                                                 \
    }                                                           \
} while(0)

#define CUDA_CHECK_NULL(call) do {                              \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "[CVT_CUDA] %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        return nullptr;                                         \
    }                                                           \
} while(0)

// ============================================================
// Double-precision atomicAdd (needed for SM < 60, but we
// provide it as a compatibility shim)
// ============================================================

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native double atomicAdd available on SM 6.0+
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ============================================================
// Context structure
// ============================================================

struct CvtCudaContext {
    // Mesh on device (uploaded once)
    double* d_vertices;      // [nb_vertices * 3]
    int*    d_triangles;     // [nb_triangles * 3]
    int     nb_vertices;
    int     nb_triangles;

    // Pre-computed triangle data
    double* d_tri_centroids; // [nb_triangles * 3] -- centroid of each triangle
    double* d_tri_areas;     // [nb_triangles]     -- area of each triangle

    // Per-iteration buffers (reused)
    double* d_seeds;         // [max_seeds * 3]
    double* d_mg;            // [max_seeds * 3]  -- mass * centroid
    double* d_m;             // [max_seeds]       -- mass
    double* d_funcval;       // [nb_triangles]    -- per-triangle energy (for reduction)
    int     max_seeds;       // allocated capacity
};

// ============================================================
// Kernel: compute triangle centroids and areas
// ============================================================

__global__ void k_compute_tri_data(
    const double* __restrict__ vertices,
    const int*    __restrict__ triangles,
    int nb_triangles,
    double* __restrict__ tri_centroids,
    double* __restrict__ tri_areas
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nb_triangles) return;

    int v0 = triangles[tid * 3 + 0];
    int v1 = triangles[tid * 3 + 1];
    int v2 = triangles[tid * 3 + 2];

    double p0x = vertices[v0*3+0], p0y = vertices[v0*3+1], p0z = vertices[v0*3+2];
    double p1x = vertices[v1*3+0], p1y = vertices[v1*3+1], p1z = vertices[v1*3+2];
    double p2x = vertices[v2*3+0], p2y = vertices[v2*3+1], p2z = vertices[v2*3+2];

    // Centroid
    tri_centroids[tid*3+0] = (p0x + p1x + p2x) / 3.0;
    tri_centroids[tid*3+1] = (p0y + p1y + p2y) / 3.0;
    tri_centroids[tid*3+2] = (p0z + p1z + p2z) / 3.0;

    // Area via cross product
    double ex1 = p1x - p0x, ey1 = p1y - p0y, ez1 = p1z - p0z;
    double ex2 = p2x - p0x, ey2 = p2y - p0y, ez2 = p2z - p0z;
    double cx = ey1*ez2 - ez1*ey2;
    double cy = ez1*ex2 - ex1*ez2;
    double cz = ex1*ey2 - ey1*ex2;
    tri_areas[tid] = 0.5 * sqrt(cx*cx + cy*cy + cz*cz);
}

// ============================================================
// Kernel: assign each triangle to nearest seed, accumulate
// area-weighted centroid (atomicAdd into per-seed buffers)
// ============================================================

__global__ void k_assign_and_accumulate(
    const double* __restrict__ tri_centroids,
    const double* __restrict__ tri_areas,
    int nb_triangles,
    const double* __restrict__ seeds,
    int nb_seeds,
    double* __restrict__ mg,   // [nb_seeds * 3]
    double* __restrict__ m     // [nb_seeds]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nb_triangles) return;

    double cx = tri_centroids[tid*3+0];
    double cy = tri_centroids[tid*3+1];
    double cz = tri_centroids[tid*3+2];
    double area = tri_areas[tid];

    if (area < 1e-30) return;

    // Find nearest seed (brute force)
    double best_dist = DBL_MAX;
    int best_seed = 0;
    for (int s = 0; s < nb_seeds; s++) {
        double dx = cx - seeds[s*3+0];
        double dy = cy - seeds[s*3+1];
        double dz = cz - seeds[s*3+2];
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_dist) {
            best_dist = d2;
            best_seed = s;
        }
    }

    // Accumulate mass * centroid and mass
    atomicAdd(&mg[best_seed*3+0], area * cx);
    atomicAdd(&mg[best_seed*3+1], area * cy);
    atomicAdd(&mg[best_seed*3+2], area * cz);
    atomicAdd(&m[best_seed], area);
}

// ============================================================
// Kernel: assign each triangle to nearest seed, accumulate
// centroid AND compute per-triangle CVT energy
//
// CVT energy for triangle T assigned to seed s:
//   E_T = area_T * (1/12) * (sum_{i<j} (pi-s).(pj-s) + sum_i ||pi-s||^2)
//       = area_T * (1/6) * (||centroid - s||^2 + (var_term))
//
// Simplified: E_T ≈ area_T * ||centroid_T - seed||^2
// (exact for point masses, good approximation for small triangles)
// ============================================================

__global__ void k_assign_accumulate_energy(
    const double* __restrict__ vertices,
    const int*    __restrict__ triangles,
    const double* __restrict__ tri_centroids,
    const double* __restrict__ tri_areas,
    int nb_triangles,
    const double* __restrict__ seeds,
    int nb_seeds,
    double* __restrict__ mg,
    double* __restrict__ m,
    double* __restrict__ funcval  // [nb_triangles] per-tri energy
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nb_triangles) return;

    double cx = tri_centroids[tid*3+0];
    double cy = tri_centroids[tid*3+1];
    double cz = tri_centroids[tid*3+2];
    double area = tri_areas[tid];

    if (area < 1e-30) {
        funcval[tid] = 0.0;
        return;
    }

    // Find nearest seed
    double best_dist = DBL_MAX;
    int best_seed = 0;
    for (int s = 0; s < nb_seeds; s++) {
        double dx = cx - seeds[s*3+0];
        double dy = cy - seeds[s*3+1];
        double dz = cz - seeds[s*3+2];
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_dist) {
            best_dist = d2;
            best_seed = s;
        }
    }

    // Accumulate centroid
    atomicAdd(&mg[best_seed*3+0], area * cx);
    atomicAdd(&mg[best_seed*3+1], area * cy);
    atomicAdd(&mg[best_seed*3+2], area * cz);
    atomicAdd(&m[best_seed], area);

    // Compute per-triangle energy using exact quadrature
    // For triangle with vertices p0,p1,p2 and seed s:
    // integral ||x-s||^2 dA = area/6 * (sum_i ||pi-s||^2 + sum_{i<j} (pi-s).(pj-s))
    int v0 = triangles[tid*3+0];
    int v1 = triangles[tid*3+1];
    int v2 = triangles[tid*3+2];
    double sx = seeds[best_seed*3+0];
    double sy = seeds[best_seed*3+1];
    double sz = seeds[best_seed*3+2];

    double d0x = vertices[v0*3+0]-sx, d0y = vertices[v0*3+1]-sy, d0z = vertices[v0*3+2]-sz;
    double d1x = vertices[v1*3+0]-sx, d1y = vertices[v1*3+1]-sy, d1z = vertices[v1*3+2]-sz;
    double d2x = vertices[v2*3+0]-sx, d2y = vertices[v2*3+1]-sy, d2z = vertices[v2*3+2]-sz;

    double sq0 = d0x*d0x + d0y*d0y + d0z*d0z;
    double sq1 = d1x*d1x + d1y*d1y + d1z*d1z;
    double sq2 = d2x*d2x + d2y*d2y + d2z*d2z;
    double dot01 = d0x*d1x + d0y*d1y + d0z*d1z;
    double dot02 = d0x*d2x + d0y*d2y + d0z*d2z;
    double dot12 = d1x*d2x + d1y*d2y + d1z*d2z;

    funcval[tid] = area / 6.0 * (sq0 + sq1 + sq2 + dot01 + dot02 + dot12);
}

// ============================================================
// Kernel: brute-force nearest neighbor
// ============================================================

__global__ void k_nearest_neighbor(
    const double* __restrict__ queries,
    int nb_queries,
    const double* __restrict__ seeds,
    int nb_seeds,
    int dim,
    int* __restrict__ out_indices
) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= nb_queries) return;

    const double* q = queries + qid * dim;
    double best_dist = DBL_MAX;
    int best_idx = 0;

    for (int s = 0; s < nb_seeds; s++) {
        const double* p = seeds + s * dim;
        double d2 = 0.0;
        for (int d = 0; d < dim; d++) {
            double diff = q[d] - p[d];
            d2 += diff * diff;
        }
        if (d2 < best_dist) {
            best_dist = d2;
            best_idx = s;
        }
    }
    out_indices[qid] = best_idx;
}

// ============================================================
// Kernel: point update (normalize mg/m)
// ============================================================

__global__ void k_update_points(
    const double* __restrict__ mg,
    const double* __restrict__ m,
    int dim, int nb_seeds,
    const int* __restrict__ locked,       // can be NULL
    const double* __restrict__ old_seeds,  // can be NULL
    double* __restrict__ out_seeds
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= nb_seeds) return;

    // If locked, keep old position
    if (locked && locked[sid]) {
        for (int d = 0; d < dim; d++) {
            out_seeds[sid*dim+d] = old_seeds[sid*dim+d];
        }
        return;
    }

    double mass = m[sid];
    if (mass > 1e-30) {
        double inv_m = 1.0 / mass;
        for (int d = 0; d < dim; d++) {
            out_seeds[sid*dim+d] = mg[sid*dim+d] * inv_m;
        }
    } else {
        // Zero-mass cell: keep old position
        if (old_seeds) {
            for (int d = 0; d < dim; d++) {
                out_seeds[sid*dim+d] = old_seeds[sid*dim+d];
            }
        }
    }
}

// ============================================================
// Host: parallel reduction for sum of double array
// (simple two-pass: GPU partial sums + CPU final sum)
// ============================================================

static double gpu_sum_doubles(const double* d_arr, int n) {
    // For simplicity, copy to host and sum
    // (for large n, a proper GPU reduction would be better)
    double* h_arr = new double[n];
    cudaMemcpy(h_arr, d_arr, n * sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += h_arr[i];
    delete[] h_arr;
    return sum;
}

// ============================================================
// Context: ensure seed buffers are large enough
// ============================================================

static void ensure_seed_capacity(CvtCudaContext* ctx, int nb_seeds) {
    if (nb_seeds <= ctx->max_seeds) return;

    // Free old
    if (ctx->d_seeds)   cudaFree(ctx->d_seeds);
    if (ctx->d_mg)      cudaFree(ctx->d_mg);
    if (ctx->d_m)       cudaFree(ctx->d_m);

    ctx->max_seeds = nb_seeds + 1024; // headroom
    cudaMalloc(&ctx->d_seeds, ctx->max_seeds * 3 * sizeof(double));
    cudaMalloc(&ctx->d_mg,    ctx->max_seeds * 3 * sizeof(double));
    cudaMalloc(&ctx->d_m,     ctx->max_seeds * sizeof(double));
}

// ============================================================
// Public API implementation
// ============================================================

extern "C" {

CvtCudaContext* cvt_cuda_create_context(
    const double* vertices, int nb_vertices,
    const int*    triangles, int nb_triangles
) {
    CvtCudaContext* ctx = new CvtCudaContext();
    memset(ctx, 0, sizeof(*ctx));
    ctx->nb_vertices = nb_vertices;
    ctx->nb_triangles = nb_triangles;

    // Upload mesh
    CUDA_CHECK_NULL(cudaMalloc(&ctx->d_vertices, nb_vertices * 3 * sizeof(double)));
    CUDA_CHECK_NULL(cudaMemcpy(ctx->d_vertices, vertices, nb_vertices * 3 * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK_NULL(cudaMalloc(&ctx->d_triangles, nb_triangles * 3 * sizeof(int)));
    CUDA_CHECK_NULL(cudaMemcpy(ctx->d_triangles, triangles, nb_triangles * 3 * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate and compute triangle data
    CUDA_CHECK_NULL(cudaMalloc(&ctx->d_tri_centroids, nb_triangles * 3 * sizeof(double)));
    CUDA_CHECK_NULL(cudaMalloc(&ctx->d_tri_areas, nb_triangles * sizeof(double)));
    CUDA_CHECK_NULL(cudaMalloc(&ctx->d_funcval, nb_triangles * sizeof(double)));

    int block = 256;
    int grid = (nb_triangles + block - 1) / block;
    k_compute_tri_data<<<grid, block>>>(
        ctx->d_vertices, ctx->d_triangles, nb_triangles,
        ctx->d_tri_centroids, ctx->d_tri_areas
    );
    CUDA_CHECK_NULL(cudaDeviceSynchronize());

    printf("[CVT_CUDA] Context created: %d vertices, %d triangles\n",
           nb_vertices, nb_triangles);

    return ctx;
}

void cvt_cuda_destroy_context(CvtCudaContext* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_vertices);
    cudaFree(ctx->d_triangles);
    cudaFree(ctx->d_tri_centroids);
    cudaFree(ctx->d_tri_areas);
    cudaFree(ctx->d_funcval);
    if (ctx->d_seeds) cudaFree(ctx->d_seeds);
    if (ctx->d_mg)    cudaFree(ctx->d_mg);
    if (ctx->d_m)     cudaFree(ctx->d_m);
    delete ctx;
}

void cvt_cuda_compute_centroids(
    CvtCudaContext* ctx,
    const double* seeds, int dim, int nb_seeds,
    double* out_mg, double* out_m
) {
    if (!ctx || dim != 3) return; // Only 3D for now

    ensure_seed_capacity(ctx, nb_seeds);

    // Upload seeds
    CUDA_CHECK(cudaMemcpy(ctx->d_seeds, seeds, nb_seeds * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // Zero accumulators
    CUDA_CHECK(cudaMemset(ctx->d_mg, 0, nb_seeds * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(ctx->d_m,  0, nb_seeds * sizeof(double)));

    // Launch assign + accumulate
    int block = 256;
    int grid = (ctx->nb_triangles + block - 1) / block;
    k_assign_and_accumulate<<<grid, block>>>(
        ctx->d_tri_centroids, ctx->d_tri_areas, ctx->nb_triangles,
        ctx->d_seeds, nb_seeds,
        ctx->d_mg, ctx->d_m
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    CUDA_CHECK(cudaMemcpy(out_mg, ctx->d_mg, nb_seeds * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_m,  ctx->d_m,  nb_seeds * sizeof(double), cudaMemcpyDeviceToHost));
}

void cvt_cuda_compute_funcgrad(
    CvtCudaContext* ctx,
    const double* seeds, int dim, int nb_seeds,
    double* out_f, double* out_g
) {
    if (!ctx || dim != 3) return;

    ensure_seed_capacity(ctx, nb_seeds);

    // Upload seeds
    CUDA_CHECK(cudaMemcpy(ctx->d_seeds, seeds, nb_seeds * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // Zero accumulators
    CUDA_CHECK(cudaMemset(ctx->d_mg, 0, nb_seeds * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(ctx->d_m,  0, nb_seeds * sizeof(double)));

    // Launch combined assign + accumulate + energy
    int block = 256;
    int grid = (ctx->nb_triangles + block - 1) / block;
    k_assign_accumulate_energy<<<grid, block>>>(
        ctx->d_vertices, ctx->d_triangles,
        ctx->d_tri_centroids, ctx->d_tri_areas, ctx->nb_triangles,
        ctx->d_seeds, nb_seeds,
        ctx->d_mg, ctx->d_m, ctx->d_funcval
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum energy on host (could use thrust::reduce for large meshes)
    *out_f = gpu_sum_doubles(ctx->d_funcval, ctx->nb_triangles);

    // Download mg and m to compute gradient
    double* h_mg = new double[nb_seeds * 3];
    double* h_m  = new double[nb_seeds];
    CUDA_CHECK(cudaMemcpy(h_mg, ctx->d_mg, nb_seeds * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m,  ctx->d_m,  nb_seeds * sizeof(double), cudaMemcpyDeviceToHost));

    // Gradient: g_i = 2 * (m_i * seed_i - mg_i)
    for (int i = 0; i < nb_seeds; i++) {
        for (int d = 0; d < 3; d++) {
            out_g[i*3+d] = 2.0 * (h_m[i] * seeds[i*3+d] - h_mg[i*3+d]);
        }
    }

    delete[] h_mg;
    delete[] h_m;
}

void cvt_cuda_nearest_neighbor(
    const double* queries, int nb_queries,
    const double* seeds, int nb_seeds,
    int dim,
    int* out_indices
) {
    double* d_queries = nullptr;
    double* d_seeds = nullptr;
    int* d_indices = nullptr;

    cudaMalloc(&d_queries, nb_queries * dim * sizeof(double));
    cudaMalloc(&d_seeds, nb_seeds * dim * sizeof(double));
    cudaMalloc(&d_indices, nb_queries * sizeof(int));

    cudaMemcpy(d_queries, queries, nb_queries * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seeds, seeds, nb_seeds * dim * sizeof(double), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (nb_queries + block - 1) / block;
    k_nearest_neighbor<<<grid, block>>>(
        d_queries, nb_queries, d_seeds, nb_seeds, dim, d_indices
    );
    cudaDeviceSynchronize();

    cudaMemcpy(out_indices, d_indices, nb_queries * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_queries);
    cudaFree(d_seeds);
    cudaFree(d_indices);
}

void cvt_cuda_update_points(
    const double* mg, const double* m,
    int dim, int nb_seeds,
    const int* locked,
    const double* old_seeds,
    double* out_seeds
) {
    double* d_mg = nullptr;
    double* d_m = nullptr;
    int* d_locked = nullptr;
    double* d_old = nullptr;
    double* d_out = nullptr;

    cudaMalloc(&d_mg, nb_seeds * dim * sizeof(double));
    cudaMalloc(&d_m,  nb_seeds * sizeof(double));
    cudaMalloc(&d_out, nb_seeds * dim * sizeof(double));

    cudaMemcpy(d_mg, mg, nb_seeds * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m,  m,  nb_seeds * sizeof(double), cudaMemcpyHostToDevice);

    if (locked) {
        cudaMalloc(&d_locked, nb_seeds * sizeof(int));
        cudaMemcpy(d_locked, locked, nb_seeds * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (old_seeds) {
        cudaMalloc(&d_old, nb_seeds * dim * sizeof(double));
        cudaMemcpy(d_old, old_seeds, nb_seeds * dim * sizeof(double), cudaMemcpyHostToDevice);
    }

    int block = 256;
    int grid = (nb_seeds + block - 1) / block;
    k_update_points<<<grid, block>>>(d_mg, d_m, dim, nb_seeds, d_locked, d_old, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(out_seeds, d_out, nb_seeds * dim * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_mg);
    cudaFree(d_m);
    cudaFree(d_out);
    if (d_locked) cudaFree(d_locked);
    if (d_old)    cudaFree(d_old);
}

// ============================================================
// Kernel: ray-triangle intersection (Moller-Trumbore)
// Returns parametric t along ray, or -1 if no hit.
// ============================================================

__device__ double ray_triangle_intersect(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double v0x, double v0y, double v0z,
    double v1x, double v1y, double v1z,
    double v2x, double v2y, double v2z,
    double* out_px, double* out_py, double* out_pz
) {
    const double EPSILON = 1e-12;
    double e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    double e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    double hx = dy*e2z - dz*e2y;
    double hy = dz*e2x - dx*e2z;
    double hz = dx*e2y - dy*e2x;

    double a = e1x*hx + e1y*hy + e1z*hz;
    if (a > -EPSILON && a < EPSILON) return -1.0;

    double f = 1.0 / a;
    double sx = ox - v0x, sy = oy - v0y, sz = oz - v0z;
    double u = f * (sx*hx + sy*hy + sz*hz);
    if (u < 0.0 || u > 1.0) return -1.0;

    double qx = sy*e1z - sz*e1y;
    double qy = sz*e1x - sx*e1z;
    double qz = sx*e1y - sy*e1x;
    double v = f * (dx*qx + dy*qy + dz*qz);
    if (v < 0.0 || u + v > 1.0) return -1.0;

    double t = f * (e2x*qx + e2y*qy + e2z*qz);
    if (t > EPSILON) {
        *out_px = ox + t * dx;
        *out_py = oy + t * dy;
        *out_pz = oz + t * dz;
        return t;
    }
    return -1.0;
}

// ============================================================
// Kernel: for each query ray, find nearest intersection with
// any reference triangle (brute force, bidirectional)
// ============================================================

__global__ void k_ray_mesh_nearest(
    const double* __restrict__ ref_verts,
    const int*    __restrict__ ref_tris,
    int nb_ref_tris,
    const double* __restrict__ query_points,
    const double* __restrict__ query_dirs,
    const double* __restrict__ max_dists,
    int nb_queries,
    double* __restrict__ out_nearest,
    int*    __restrict__ out_hit
) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= nb_queries) return;

    double ox = query_points[qid*3+0];
    double oy = query_points[qid*3+1];
    double oz = query_points[qid*3+2];
    double dx = query_dirs[qid*3+0];
    double dy = query_dirs[qid*3+1];
    double dz = query_dirs[qid*3+2];
    double max_d = max_dists[qid];

    double best_dist2 = max_d * max_d;
    double best_px = ox, best_py = oy, best_pz = oz;
    int hit = 0;

    // Test both directions
    for (int sign = 0; sign < 2; sign++) {
        double sdx = (sign == 0) ? dx : -dx;
        double sdy = (sign == 0) ? dy : -dy;
        double sdz = (sign == 0) ? dz : -dz;

        for (int f = 0; f < nb_ref_tris; f++) {
            int i0 = ref_tris[f*3+0];
            int i1 = ref_tris[f*3+1];
            int i2 = ref_tris[f*3+2];

            double px, py, pz;
            double t = ray_triangle_intersect(
                ox, oy, oz, sdx, sdy, sdz,
                ref_verts[i0*3+0], ref_verts[i0*3+1], ref_verts[i0*3+2],
                ref_verts[i1*3+0], ref_verts[i1*3+1], ref_verts[i1*3+2],
                ref_verts[i2*3+0], ref_verts[i2*3+1], ref_verts[i2*3+2],
                &px, &py, &pz
            );
            if (t > 0.0) {
                double ddx = px - ox, ddy = py - oy, ddz = pz - oz;
                double d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_px = px;
                    best_py = py;
                    best_pz = pz;
                    hit = 1;
                }
            }
        }
    }

    out_nearest[qid*3+0] = best_px;
    out_nearest[qid*3+1] = best_py;
    out_nearest[qid*3+2] = best_pz;
    out_hit[qid] = hit;
}

// ============================================================
// Kernel: compute area-weighted vertex normals
// ============================================================

__global__ void k_vertex_normals_accumulate(
    const double* __restrict__ vertices,
    const int*    __restrict__ triangles,
    int nb_tris,
    double* __restrict__ normals  // [nb_verts * 3], must be zeroed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nb_tris) return;

    int v0 = triangles[tid*3+0];
    int v1 = triangles[tid*3+1];
    int v2 = triangles[tid*3+2];

    double p0x = vertices[v0*3+0], p0y = vertices[v0*3+1], p0z = vertices[v0*3+2];
    double p1x = vertices[v1*3+0], p1y = vertices[v1*3+1], p1z = vertices[v1*3+2];
    double p2x = vertices[v2*3+0], p2y = vertices[v2*3+1], p2z = vertices[v2*3+2];

    double e1x = p1x-p0x, e1y = p1y-p0y, e1z = p1z-p0z;
    double e2x = p2x-p0x, e2y = p2y-p0y, e2z = p2z-p0z;

    // Cross product (not normalized = area-weighted)
    double nx = e1y*e2z - e1z*e2y;
    double ny = e1z*e2x - e1x*e2z;
    double nz = e1x*e2y - e1y*e2x;

    // Accumulate to all 3 vertices
    atomicAdd(&normals[v0*3+0], nx);
    atomicAdd(&normals[v0*3+1], ny);
    atomicAdd(&normals[v0*3+2], nz);
    atomicAdd(&normals[v1*3+0], nx);
    atomicAdd(&normals[v1*3+1], ny);
    atomicAdd(&normals[v1*3+2], nz);
    atomicAdd(&normals[v2*3+0], nx);
    atomicAdd(&normals[v2*3+1], ny);
    atomicAdd(&normals[v2*3+2], nz);
}

__global__ void k_normalize_normals(
    double* __restrict__ normals,
    int nb_verts
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= nb_verts) return;

    double nx = normals[vid*3+0];
    double ny = normals[vid*3+1];
    double nz = normals[vid*3+2];
    double len = sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 1e-30) {
        normals[vid*3+0] = nx / len;
        normals[vid*3+1] = ny / len;
        normals[vid*3+2] = nz / len;
    }
}

// ============================================================
// Host API: ray-mesh nearest
// ============================================================

void cvt_cuda_ray_mesh_nearest(
    const double* ref_vertices, const int* ref_triangles,
    int nb_ref_verts, int nb_ref_tris,
    const double* query_points, const double* query_dirs,
    const double* max_dists,
    int nb_queries,
    double* out_nearest, int* out_hit
) {
    double* d_ref_v = nullptr;
    int*    d_ref_t = nullptr;
    double* d_qp = nullptr;
    double* d_qd = nullptr;
    double* d_md = nullptr;
    double* d_out = nullptr;
    int*    d_hit = nullptr;

    cudaMalloc(&d_ref_v, nb_ref_verts * 3 * sizeof(double));
    cudaMalloc(&d_ref_t, nb_ref_tris * 3 * sizeof(int));
    cudaMalloc(&d_qp, nb_queries * 3 * sizeof(double));
    cudaMalloc(&d_qd, nb_queries * 3 * sizeof(double));
    cudaMalloc(&d_md, nb_queries * sizeof(double));
    cudaMalloc(&d_out, nb_queries * 3 * sizeof(double));
    cudaMalloc(&d_hit, nb_queries * sizeof(int));

    cudaMemcpy(d_ref_v, ref_vertices, nb_ref_verts * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_t, ref_triangles, nb_ref_tris * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qp, query_points, nb_queries * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qd, query_dirs, nb_queries * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_md, max_dists, nb_queries * sizeof(double), cudaMemcpyHostToDevice);

    int block = 128;  // Fewer threads per block since each does O(T) work
    int grid = (nb_queries + block - 1) / block;
    k_ray_mesh_nearest<<<grid, block>>>(
        d_ref_v, d_ref_t, nb_ref_tris,
        d_qp, d_qd, d_md, nb_queries,
        d_out, d_hit
    );
    cudaDeviceSynchronize();

    cudaMemcpy(out_nearest, d_out, nb_queries * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_hit, d_hit, nb_queries * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_ref_v);
    cudaFree(d_ref_t);
    cudaFree(d_qp);
    cudaFree(d_qd);
    cudaFree(d_md);
    cudaFree(d_out);
    cudaFree(d_hit);
}

// ============================================================
// Host API: vertex normals
// ============================================================

void cvt_cuda_compute_vertex_normals(
    const double* vertices, const int* triangles,
    int nb_verts, int nb_tris,
    double* out_normals
) {
    double* d_verts = nullptr;
    int*    d_tris = nullptr;
    double* d_normals = nullptr;

    cudaMalloc(&d_verts, nb_verts * 3 * sizeof(double));
    cudaMalloc(&d_tris, nb_tris * 3 * sizeof(int));
    cudaMalloc(&d_normals, nb_verts * 3 * sizeof(double));

    cudaMemcpy(d_verts, vertices, nb_verts * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tris, triangles, nb_tris * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_normals, 0, nb_verts * 3 * sizeof(double));

    int block = 256;
    int grid1 = (nb_tris + block - 1) / block;
    k_vertex_normals_accumulate<<<grid1, block>>>(d_verts, d_tris, nb_tris, d_normals);

    int grid2 = (nb_verts + block - 1) / block;
    k_normalize_normals<<<grid2, block>>>(d_normals, nb_verts);

    cudaDeviceSynchronize();
    cudaMemcpy(out_normals, d_normals, nb_verts * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_verts);
    cudaFree(d_tris);
    cudaFree(d_normals);
}

} // extern "C"
