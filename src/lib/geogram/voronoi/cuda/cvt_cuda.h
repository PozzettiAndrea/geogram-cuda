#ifndef GEOGRAM_VORONOI_CVT_CUDA_H
#define GEOGRAM_VORONOI_CVT_CUDA_H

// ============================================================
// CUDA-accelerated kernels for Centroidal Voronoi Tessellation
//
// Provides GPU implementations for the hot inner loops of CVT:
//   1. Lloyd centroid computation (per-triangle -> per-seed reduction)
//   2. CVT function + gradient evaluation
//   3. Nearest-neighbor seed assignment
//   4. Point update (centroid normalization)
//
// Strategy: each mesh triangle is assigned to its nearest Voronoi
// seed. The triangle's area-weighted centroid contribution is
// atomically accumulated into the seed's centroid buffer. This is
// an approximation when a triangle spans multiple Voronoi cells,
// but converges to the exact solution as Lloyd iterations proceed
// (triangles become smaller relative to cells). For meshes with
// many more triangles than seeds, this gives excellent parallelism.
//
// For exact RVD clipping, fall back to CPU (strategy=cpu).
// ============================================================

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Device context (opaque handle to GPU-resident data)
// ============================================================

struct CvtCudaContext;

// Create context: uploads mesh triangles to GPU.
//   vertices:  [nb_vertices * 3] doubles (x,y,z per vertex)
//   triangles: [nb_triangles * 3] ints (v0,v1,v2 per triangle)
// Returns nullptr on failure.
CvtCudaContext* cvt_cuda_create_context(
    const double* vertices, int nb_vertices,
    const int*    triangles, int nb_triangles
);

// Destroy context and free GPU memory.
void cvt_cuda_destroy_context(CvtCudaContext* ctx);

// ============================================================
// Lloyd centroid computation (GPU)
//
// For each triangle, find nearest seed, accumulate area-weighted
// centroid into per-seed buffers. Then normalize.
//
//   seeds:       [nb_seeds * dim] doubles (current seed positions, host)
//   dim:         coordinate dimension (typically 3)
//   nb_seeds:    number of Voronoi seeds
//   out_mg:      [nb_seeds * dim] doubles (mass * centroid, output, host)
//   out_m:       [nb_seeds] doubles (mass, output, host)
//
// After call: new_seed[i] = out_mg[i*dim..] / out_m[i]
// ============================================================

void cvt_cuda_compute_centroids(
    CvtCudaContext* ctx,
    const double* seeds, int dim, int nb_seeds,
    double* out_mg, double* out_m
);

// ============================================================
// CVT function + gradient evaluation (GPU)
//
// Computes the quantization noise power (CVT energy):
//   E = sum_i sum_{T in cell_i} integral_T ||x - seed_i||^2 dA
//
// Approximated per-triangle as:
//   E_T = area_T * (1/12)(||p0-s||^2 + ||p1-s||^2 + ||p2-s||^2
//         + (p0-s).(p1-s) + (p0-s).(p2-s) + (p1-s).(p2-s))
//
// Gradient w.r.t. seed_i:
//   g_i = 2 * (m_i * seed_i - mg_i)
//
//   seeds:     [nb_seeds * dim] doubles (current positions, host)
//   dim:       coordinate dimension
//   nb_seeds:  number of seeds
//   out_f:     scalar output (objective function value)
//   out_g:     [nb_seeds * dim] doubles (gradient, output, host)
// ============================================================

void cvt_cuda_compute_funcgrad(
    CvtCudaContext* ctx,
    const double* seeds, int dim, int nb_seeds,
    double* out_f, double* out_g
);

// ============================================================
// Nearest-neighbor assignment (GPU)
//
// For each query point, find the index of the nearest seed.
// Uses brute-force O(Q*S) -- effective for up to ~100K seeds
// on modern GPUs.
//
//   queries:     [nb_queries * dim] doubles
//   nb_queries:  number of query points
//   seeds:       [nb_seeds * dim] doubles
//   nb_seeds:    number of seeds
//   dim:         coordinate dimension
//   out_indices: [nb_queries] ints (nearest seed index per query)
// ============================================================

void cvt_cuda_nearest_neighbor(
    const double* queries, int nb_queries,
    const double* seeds, int nb_seeds,
    int dim,
    int* out_indices
);

// ============================================================
// Point update (GPU) -- normalize mg/m to get new seed positions
//
//   mg:        [nb_seeds * dim] doubles (mass * centroid)
//   m:         [nb_seeds] doubles (mass)
//   dim:       coordinate dimension
//   nb_seeds:  number of seeds
//   locked:    [nb_seeds] ints (1 = locked, 0 = free). Can be NULL.
//   old_seeds: [nb_seeds * dim] doubles (previous positions, for locked)
//   out_seeds: [nb_seeds * dim] doubles (new positions, output)
// ============================================================

void cvt_cuda_update_points(
    const double* mg, const double* m,
    int dim, int nb_seeds,
    const int* locked,
    const double* old_seeds,
    double* out_seeds
);

// ============================================================
// GPU surface sampling (cuRAND + prefix scan)
//
// Generates nb_samples random points on a triangle mesh surface,
// weighted by triangle area. Much faster than CPU for large
// sample counts.
//
//   h_vertices:   [nb_vertices * 3] doubles (host)
//   h_triangles:  [nb_triangles * 3] ints (host)
//   h_tri_areas:  [nb_triangles] doubles (pre-computed areas, host)
//   nb_samples:   number of samples to generate
//   seed:         cuRAND seed for reproducibility
//   out_points:   [nb_samples * 3] doubles (output, host)
// ============================================================

void cvt_cuda_generate_surface_samples(
    const double* h_vertices, int nb_vertices,
    const int* h_triangles, int nb_triangles,
    const double* h_tri_areas,
    int nb_samples,
    unsigned long long seed,
    double* out_points
);

// ============================================================
// Surface adjust: GPU-accelerated nearest-point-on-surface
//
// For each query point + direction, find the nearest intersection
// with the reference mesh (bidirectional ray). This is the hot
// loop of mesh_adjust_surface().
//
// Uses brute-force ray-triangle intersection over all reference
// triangles. O(Q * T) but massively parallel on GPU.
//
//   ref_vertices:  [nb_ref_verts * 3] doubles
//   ref_triangles: [nb_ref_tris * 3] ints
//   nb_ref_verts:  number of reference mesh vertices
//   nb_ref_tris:   number of reference mesh triangles
//   query_points:  [nb_queries * 3] doubles (ray origins)
//   query_dirs:    [nb_queries * 3] doubles (ray directions)
//   max_dists:     [nb_queries] doubles (max distance per query)
//   nb_queries:    number of query rays
//   out_nearest:   [nb_queries * 3] doubles (nearest point found)
//   out_hit:       [nb_queries] ints (1 if hit, 0 if not)
// ============================================================

void cvt_cuda_ray_mesh_nearest(
    const double* ref_vertices, const int* ref_triangles,
    int nb_ref_verts, int nb_ref_tris,
    const double* query_points, const double* query_dirs,
    const double* max_dists,
    int nb_queries,
    double* out_nearest, int* out_hit
);

// ============================================================
// Compute vertex normals from face normals (area-weighted)
//
//   vertices:   [nb_verts * 3] doubles
//   triangles:  [nb_tris * 3] ints
//   nb_verts:   number of vertices
//   nb_tris:    number of triangles
//   out_normals: [nb_verts * 3] doubles (output, area-weighted normals)
// ============================================================

void cvt_cuda_compute_vertex_normals(
    const double* vertices, const int* triangles,
    int nb_verts, int nb_tris,
    double* out_normals
);

#ifdef __cplusplus
}
#endif

#endif // GEOGRAM_VORONOI_CVT_CUDA_H
