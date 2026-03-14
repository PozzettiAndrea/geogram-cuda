// ============================================================
// CUDA kernels for GPU-accelerated surface sampling
//
// Generates random points on a triangle mesh surface, weighted
// by triangle area. Uses cuRAND for random number generation
// and thrust::inclusive_scan for the area CDF.
//
// Algorithm:
//   1. Compute per-triangle areas (reuse from CvtCudaContext)
//   2. Prefix sum → cumulative distribution function (CDF)
//   3. For each sample: binary search CDF → triangle index,
//      generate barycentric coords via cuRAND → 3D point
// ============================================================

#include "cvt_cuda.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <cstdio>
#include <cstring>

// Forward declaration of context (defined in cvt_cuda.cu)
struct CvtCudaContext;

// We need access to the context fields. Since CvtCudaContext is defined
// in cvt_cuda.cu, we re-declare the fields we need here.
// This is safe because we only read from the context.
extern "C" {

// ============================================================
// Kernel: initialize cuRAND states
// ============================================================

__global__ void k_init_curand(
    curandState* __restrict__ states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

// ============================================================
// Kernel: generate surface samples
//
// Each thread generates one sample point:
// 1. Draw uniform u in [0, total_area]
// 2. Binary search CDF to find triangle
// 3. Draw (r1, r2) for barycentric coords
// 4. Compute 3D point on triangle
// ============================================================

__global__ void k_generate_surface_samples(
    const double* __restrict__ cdf,        // [nb_triangles] inclusive prefix sum of areas
    const double* __restrict__ vertices,   // [nb_vertices * 3]
    const int*    __restrict__ triangles,  // [nb_triangles * 3]
    int nb_triangles,
    double total_area,
    curandState* __restrict__ rand_states,
    int nb_samples,
    double* __restrict__ out_points        // [nb_samples * 3]
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= nb_samples) return;

    curandState local_state = rand_states[sid];

    // 1. Draw uniform in [0, total_area)
    double u = curand_uniform_double(&local_state) * total_area;

    // 2. Binary search CDF to find triangle index
    // CDF[i] = sum of areas[0..i], so we want smallest i where CDF[i] >= u
    int lo = 0, hi = nb_triangles - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int tri_idx = lo;

    // 3. Generate random barycentric coordinates
    // Use the standard sqrt method for uniform sampling in triangle
    double r1 = curand_uniform_double(&local_state);
    double r2 = curand_uniform_double(&local_state);
    double sqrt_r1 = sqrt(r1);
    double b0 = 1.0 - sqrt_r1;
    double b1 = r2 * sqrt_r1;
    double b2 = 1.0 - b0 - b1;

    // 4. Compute 3D point
    int v0 = triangles[tri_idx * 3 + 0];
    int v1 = triangles[tri_idx * 3 + 1];
    int v2 = triangles[tri_idx * 3 + 2];

    out_points[sid * 3 + 0] = b0 * vertices[v0*3+0] + b1 * vertices[v1*3+0] + b2 * vertices[v2*3+0];
    out_points[sid * 3 + 1] = b0 * vertices[v0*3+1] + b1 * vertices[v1*3+1] + b2 * vertices[v2*3+1];
    out_points[sid * 3 + 2] = b0 * vertices[v0*3+2] + b1 * vertices[v1*3+2] + b2 * vertices[v2*3+2];

    rand_states[sid] = local_state;
}

// ============================================================
// Host API
// ============================================================

void cvt_cuda_generate_surface_samples(
    const double* h_vertices, int nb_vertices,
    const int* h_triangles, int nb_triangles,
    const double* h_tri_areas,  // pre-computed triangle areas (host), or NULL
    int nb_samples,
    unsigned long long seed,
    double* out_points
) {
    // Upload mesh data
    double* d_vertices = nullptr;
    int* d_triangles = nullptr;
    double* d_areas = nullptr;
    double* d_cdf = nullptr;
    curandState* d_rand_states = nullptr;
    double* d_out = nullptr;

    cudaMalloc(&d_vertices, nb_vertices * 3 * sizeof(double));
    cudaMalloc(&d_triangles, nb_triangles * 3 * sizeof(int));
    cudaMalloc(&d_areas, nb_triangles * sizeof(double));
    cudaMalloc(&d_cdf, nb_triangles * sizeof(double));
    cudaMalloc(&d_rand_states, nb_samples * sizeof(curandState));
    cudaMalloc(&d_out, nb_samples * 3 * sizeof(double));

    cudaMemcpy(d_vertices, h_vertices, nb_vertices * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, h_triangles, nb_triangles * 3 * sizeof(int), cudaMemcpyHostToDevice);

    if (h_tri_areas) {
        cudaMemcpy(d_areas, h_tri_areas, nb_triangles * sizeof(double), cudaMemcpyHostToDevice);
    } else {
        // Compute areas on GPU (reuse kernel from cvt_cuda.cu - but we'd need it here)
        // For now, require pre-computed areas
        fprintf(stderr, "[SAMPLING_CUDA] ERROR: h_tri_areas must not be NULL\n");
        cudaFree(d_vertices); cudaFree(d_triangles); cudaFree(d_areas);
        cudaFree(d_cdf); cudaFree(d_rand_states); cudaFree(d_out);
        return;
    }

    // Prefix sum → CDF
    thrust::device_ptr<double> areas_ptr(d_areas);
    thrust::device_ptr<double> cdf_ptr(d_cdf);
    thrust::inclusive_scan(areas_ptr, areas_ptr + nb_triangles, cdf_ptr);

    // Get total area (last element of CDF)
    double total_area = 0.0;
    cudaMemcpy(&total_area, d_cdf + nb_triangles - 1, sizeof(double), cudaMemcpyDeviceToHost);

    if (total_area < 1e-30) {
        fprintf(stderr, "[SAMPLING_CUDA] ERROR: total mesh area is zero\n");
        cudaFree(d_vertices); cudaFree(d_triangles); cudaFree(d_areas);
        cudaFree(d_cdf); cudaFree(d_rand_states); cudaFree(d_out);
        return;
    }

    // Initialize cuRAND states
    int block = 256;
    int grid_rand = (nb_samples + block - 1) / block;
    k_init_curand<<<grid_rand, block>>>(d_rand_states, seed, nb_samples);
    cudaDeviceSynchronize();

    // Generate samples
    k_generate_surface_samples<<<grid_rand, block>>>(
        d_cdf, d_vertices, d_triangles, nb_triangles,
        total_area, d_rand_states, nb_samples, d_out
    );
    cudaDeviceSynchronize();

    // Download
    cudaMemcpy(out_points, d_out, nb_samples * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    printf("[SAMPLING_CUDA] Generated %d samples on %d triangles (area=%.6f)\n",
           nb_samples, nb_triangles, total_area);

    // Cleanup
    cudaFree(d_vertices);
    cudaFree(d_triangles);
    cudaFree(d_areas);
    cudaFree(d_cdf);
    cudaFree(d_rand_states);
    cudaFree(d_out);
}

} // extern "C"
