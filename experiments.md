# Geogram-CUDA: Voronoi Remeshing GPU Acceleration

## Overview

CUDA parallelization of geogram's CVT (Centroidal Voronoi Tessellation) remeshing
pipeline, with config-driven strategy optionality and checkpoint/resume support.

Part of the `cudageom` project family (QuadriFlow-cuda, instant-meshes-cuda,
quadwild-bimdf-cuda, geogram-cuda, pmp-library-cuda, mmg-cuda).

## Architecture

### Pipeline Stages

The remeshing pipeline is decomposed into 6 checkpointable stages:

```
post-load     →  Mesh loaded + preprocessed (gradation, anisotropy)
post-sample   →  Initial random sampling computed (Voronoi seeds)
post-lloyd    →  Lloyd relaxation iterations done (centroid-based)
post-newton   →  Newton/HLBFGS iterations done (gradient-based)
post-extract  →  RDT surface mesh extracted
post-adjust   →  Surface adjustment (least-squares fitting to input)
```

Each stage boundary is a valid checkpoint point. State saved:
- Seed point positions (the main evolving data)
- CVT configuration (dimension, iteration counts, flags)
- Strategy configuration
- Output mesh (after extraction)

### Strategy Axes

Three components can independently run on CPU or CUDA:

| Flag | CPU (default) | CUDA |
|------|---------------|------|
| `-lloyd-strategy` | Exact RVD clipping (geogram's generic_RVD) | Per-triangle nearest-seed assignment + atomic centroid accumulation |
| `-funcgrad-strategy` | Exact RVD integration simplices | Per-triangle energy quadrature + atomic gradient accumulation |
| `-nn-strategy` | ANN/kd-tree (geogram's Delaunay NN) | Brute-force GPU O(Q*S) |

### CUDA Kernel Design

**Core approach:** Per-triangle parallelism with atomic accumulation.

For each mesh triangle T:
1. Find nearest Voronoi seed s (brute-force NN on GPU)
2. Accumulate `area(T) * centroid(T)` into `mg[s]` via `atomicAdd`
3. Accumulate `area(T)` into `m[s]` via `atomicAdd`

This is an **approximation** of exact RVD clipping -- assigns each whole
triangle to a single Voronoi cell rather than clipping at cell boundaries.
The approximation improves as:
- More iterations proceed (triangles become small relative to cells)
- Triangle count increases relative to seed count

For exact results, use `-lloyd-strategy cpu` (default).

**Kernels in `cvt_cuda.cu`:**

| Kernel | Purpose | Parallelism |
|--------|---------|-------------|
| `k_compute_tri_data` | Pre-compute triangle centroids + areas | 1 thread/triangle |
| `k_assign_and_accumulate` | Nearest-seed + centroid accumulation | 1 thread/triangle |
| `k_assign_accumulate_energy` | Above + CVT energy quadrature | 1 thread/triangle |
| `k_nearest_neighbor` | Brute-force NN queries | 1 thread/query |
| `k_update_points` | Normalize mg/m to get new seeds | 1 thread/seed |

**Data flow per Lloyd iteration:**

```
[Host → Device]  seeds (nb_seeds * 3 doubles)
[Device]         k_assign_and_accumulate: for each triangle, find nearest seed,
                 atomicAdd centroid contribution
[Device → Host]  mg[], m[] (per-seed accumulators)
[Host]           new_seed[i] = mg[i] / m[i]
```

Mesh data (vertices, triangles, pre-computed areas/centroids) uploaded once
at context creation and reused across iterations.

### CVT Energy Quadrature

For the Newton optimization, we need the CVT objective function:

```
E = Σ_i ∫_{V_i ∩ M} ||x - s_i||² dA
```

Exact quadrature over a triangle T assigned to seed s:

```
E_T = area(T)/6 * (||p0-s||² + ||p1-s||² + ||p2-s||²
                    + (p0-s)·(p1-s) + (p0-s)·(p2-s) + (p1-s)·(p2-s))
```

Gradient: `g_i = 2 * (m_i * s_i - mg_i)`

### Checkpoint System

Binary format following QFC/IMC/QWC pattern:

- **Magic:** `GEC\0` (4 bytes)
- **Version:** 1
- **Header:** 512 bytes (stage name, strategy flags, CVT params, mesh stats, input path, timestamp, reserved padding)
- **Body:** Conditional serialization based on stage (only saves data available at that stage)
- **Extension:** `.gec`
- **Path:** `{save_dir}/{stage_name}.gec`

### File Layout

```
src/lib/geogram/voronoi/cuda/
├── cvt_cuda.h          # C interface to CUDA kernels
├── cvt_cuda.cu         # CUDA kernels (centroid, funcgrad, NN, update)
├── geo_checkpoint.h    # Checkpoint system header
├── geo_checkpoint.cpp  # Checkpoint implementation
└── geo_serialize.h     # Binary serialization templates

src/bin/vorpalite/
└── main.cpp            # Modified with staged pipeline + checkpoint CLI

CMakeLists.txt          # Added GEOGRAM_WITH_CUDA option
```

## Usage Examples

```bash
# Standard CPU remesh (unchanged behavior)
./vorpalite model.obj out.obj remesh remesh:nb_pts=30000

# CUDA Lloyd with checkpoints
./vorpalite model.obj out.obj remesh remesh:nb_pts=30000 \
    -lloyd-strategy cuda -save-all -save-dir /tmp/geo_ckpt

# Resume from post-lloyd, run only Newton
./vorpalite model.obj out.obj remesh \
    -run-from post-lloyd -run-to post-newton -save-dir /tmp/geo_ckpt

# Benchmark Lloyd stage only
./vorpalite model.obj out.obj remesh remesh:nb_pts=30000 \
    -run-from post-sample -run-to post-lloyd \
    -lloyd-strategy cuda -save-dir /tmp/geo_ckpt

# List stages
./vorpalite -list-stages
```

## Build

```bash
mkdir build && cd build

# CPU only (default)
cmake .. -DCMAKE_BUILD_TYPE=Release

# With CUDA
cmake .. -DCMAKE_BUILD_TYPE=Release -DGEOGRAM_WITH_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

make -j$(nproc)
```

## Expected Bottlenecks

Based on profiling the CPU pipeline:

1. **RVD computation** (Lloyd + Newton inner loops) -- This is the main target.
   Exact RVD clipping is inherently sequential per seed-triangle pair, but
   the approximate per-triangle approach enables massive parallelism.

2. **Delaunay construction** (`set_vertices`) -- Called every Lloyd iteration.
   Geogram already has `ParallelDelaunay3d` for CPU. GPU Delaunay is a
   research problem; not attempted here.

3. **Surface extraction** (`compute_RDT`) -- Called once. Not a bottleneck.

4. **Surface adjustment** (`mesh_adjust_surface`) -- Uses OpenNL least-squares.
   Could benefit from OpenNL's existing CUDA support (`nl:CUDA`).

## Comparison with Other cudageom Projects

| Feature | QuadriFlow | Instant Meshes | QuadWild | **Geogram** |
|---------|-----------|---------------|----------|-------------|
| Pipeline stages | 12 | 6 | 5 | **6** |
| CUDA kernels | 5 .cu files | 1 .cu file | 2 .cu files | **1 .cu file** |
| Checkpoint magic | QFC | IMC | QWC | **GEC** |
| Strategy axes | 3 (ff/subdiv/dse) | 0 | 1 (smooth) | **3 (lloyd/funcgrad/nn)** |
| Main GPU target | Max-flow + subdivision | Field optimization | Smoothing | **CVT centroids** |

## Future Work

- GPU-accelerated spatial hashing for NN (replace brute-force for >100K seeds)
- Shared memory tiling in `k_assign_and_accumulate` for seed array
- Warp-level reduction instead of atomicAdd for high-contention seeds
- GPU-side Delaunay (if research matures)
- Integration with OpenNL CUDA for the Newton Hessian solve
