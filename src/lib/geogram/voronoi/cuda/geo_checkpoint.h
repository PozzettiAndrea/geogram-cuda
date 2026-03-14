#ifndef GEOGRAM_VORONOI_GEO_CHECKPOINT_H
#define GEOGRAM_VORONOI_GEO_CHECKPOINT_H

// ============================================================
// Pipeline checkpoint system for geogram CVT remeshing
//
// Follows the QFC/IMC/QWC pattern from the cudageom project family.
// Enables saving/loading full pipeline state at stage boundaries
// for benchmarking individual stages and resume-from-checkpoint.
//
// Usage:
//   Full run with saves:
//     ./vorpalite model.obj out.obj remesh -save-all -save-dir /tmp/ckpt
//
//   Resume from stage:
//     ./vorpalite model.obj out.obj remesh -run-from post-lloyd \
//         -save-dir /tmp/ckpt
//
//   Run single stage:
//     ./vorpalite model.obj out.obj remesh -run-from post-lloyd \
//         -run-to post-newton -save-dir /tmp/ckpt
//
// Pipeline stages (in order):
//   post-load        After mesh load + preprocessing
//   post-sample      After compute_initial_sampling()
//   post-lloyd       After Lloyd_iterations()
//   post-newton      After Newton_iterations()
//   post-extract     After compute_surface() / compute_RDT()
//   post-adjust      After mesh_adjust_surface() (final)
// ============================================================

#include <string>
#include <vector>

namespace geo_ckpt {

// ============================================================
// Pipeline stages
// ============================================================

enum PipelineStage {
    STAGE_NONE = -1,
    STAGE_POST_LOAD = 0,
    STAGE_POST_SAMPLE,
    STAGE_POST_LLOYD,
    STAGE_POST_NEWTON,
    STAGE_POST_EXTRACT,
    STAGE_POST_ADJUST,
    STAGE_COUNT
};

PipelineStage stage_from_name(const char* name);
const char* stage_name(PipelineStage s);

// ============================================================
// Strategy configuration
// ============================================================

struct StrategyConfig {
    int lloyd_strategy;     // 0=cpu (exact RVD), 1=cuda (approx per-triangle)
    int funcgrad_strategy;  // 0=cpu (exact RVD), 1=cuda (approx per-triangle)
    int nn_strategy;        // 0=cpu (ANN/kd-tree), 1=cuda (brute-force)
    int adjust_strategy;    // 0=cpu (AABB ray-cast), 1=cuda (GPU LBVH ray-cast)
    int sample_strategy;    // 0=cpu (weighted random), 1=cuda (cuRAND + prefix scan)
    int extract_strategy;   // 0=cpu (exact RVD/RDT), 1=cuda (approx nearest-seed)
    int solve_strategy;     // 0=cpu (OpenNL CG), 1=cuda (OpenNL cuSPARSE)

    StrategyConfig()
        : lloyd_strategy(0), funcgrad_strategy(0), nn_strategy(0),
          adjust_strategy(0), sample_strategy(0), extract_strategy(0),
          solve_strategy(0) {}
};

// ============================================================
// Checkpoint header (binary, stored at start of each file)
// ============================================================

struct CheckpointHeader {
    char magic[4];              // "GEC\0"
    int version;                // format version (1)
    char stage[64];             // stage name

    // Strategy flags
    int lloyd_strategy;
    int funcgrad_strategy;
    int nn_strategy;
    int adjust_strategy;
    int sample_strategy;
    int extract_strategy;
    int solve_strategy;

    // CVT parameters
    int nb_points;              // number of Voronoi seeds
    int dimension;              // coordinate dimension (3 or 6)
    int nb_lloyd_iter;          // configured Lloyd iterations
    int nb_newton_iter;         // configured Newton iterations
    int newton_m;               // HLBFGS M parameter
    int volumetric;             // 0=surfacic, 1=volumetric

    // Mesh stats
    int nb_vertices;
    int nb_facets;

    // Input info
    char input_mesh[256];
    long long timestamp;

    // Padding for future use
    char reserved[128];
};

// ============================================================
// Checkpoint data bundle
//
// This struct holds all the data that gets checkpointed.
// It does NOT own the Mesh (the mesh is re-loaded from the
// input file on resume, since it's too large to checkpoint
// efficiently and doesn't change after preprocessing).
// ============================================================

struct CheckpointData {
    // CVT parameters
    int dimension;
    int nb_points;
    int nb_lloyd_iter;
    int nb_newton_iter;
    int newton_m;
    bool volumetric;
    bool use_RVC_centroids;
    bool constrained_cvt;
    bool multi_nerve;
    bool adjust;
    double adjust_max_edge_distance;
    double adjust_border_importance;

    // Strategy config
    StrategyConfig strategy;

    // Seed points (the main evolving state)
    std::vector<double> points;          // [nb_points * dimension]

    // Locked points
    std::vector<bool> point_is_locked;

    // Output mesh vertices/triangles (available after post-extract)
    std::vector<double> out_vertices;    // [nb_out_verts * 3]
    std::vector<int>    out_triangles;   // [nb_out_tris * 3]
    int nb_out_vertices;
    int nb_out_triangles;

    // Input mesh path (for re-loading on resume)
    std::string input_mesh_path;

    CheckpointData()
        : dimension(3), nb_points(0), nb_lloyd_iter(5), nb_newton_iter(30),
          newton_m(7), volumetric(false), use_RVC_centroids(true),
          constrained_cvt(false), multi_nerve(true), adjust(true),
          adjust_max_edge_distance(0.5), adjust_border_importance(2.0),
          nb_out_vertices(0), nb_out_triangles(0) {}
};

// ============================================================
// Save/Load API
// ============================================================

// Save checkpoint at given stage
void save_checkpoint(
    const CheckpointData& data,
    PipelineStage stage,
    const char* dir
);

// Load checkpoint, returns the stage that was saved
PipelineStage load_checkpoint(
    CheckpointData& data,
    const char* dir,
    PipelineStage stage
);

// Check if checkpoint file exists for a given stage
bool checkpoint_exists(const char* dir, PipelineStage stage);

// List all available stages to stdout
void list_stages();

} // namespace geo_ckpt

#endif // GEOGRAM_VORONOI_GEO_CHECKPOINT_H
