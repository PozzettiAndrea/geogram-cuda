#include "geo_checkpoint.h"
#include "geo_serialize.h"

#include <cstring>
#include <ctime>
#include <cstdio>
#include <sys/stat.h>

namespace geo_ckpt {

// ============================================================
// Stage name mapping
// ============================================================

static const char* stage_names[] = {
    "post-load",
    "post-sample",
    "post-lloyd",
    "post-newton",
    "post-extract",
    "post-adjust",
};

PipelineStage stage_from_name(const char* name) {
    for (int i = 0; i < STAGE_COUNT; ++i) {
        if (strcmp(name, stage_names[i]) == 0) return (PipelineStage)i;
    }
    return STAGE_NONE;
}

const char* stage_name(PipelineStage s) {
    if (s >= 0 && s < STAGE_COUNT) return stage_names[s];
    return "unknown";
}

void list_stages() {
    printf("Pipeline stages:\n");
    for (int s = 0; s < STAGE_COUNT; ++s) {
        printf("  %2d. %s\n", s, stage_names[s]);
    }
}

// ============================================================
// Checkpoint file path
// ============================================================

static std::string checkpoint_path(const char* dir, PipelineStage stage) {
    return std::string(dir) + "/" + stage_names[stage] + ".gec";
}

bool checkpoint_exists(const char* dir, PipelineStage stage) {
    struct stat st;
    std::string path = checkpoint_path(dir, stage);
    return stat(path.c_str(), &st) == 0;
}

// ============================================================
// Save checkpoint
// ============================================================

void save_checkpoint(
    const CheckpointData& data,
    PipelineStage stage,
    const char* dir
) {
    // Ensure directory exists
    mkdir(dir, 0755);

    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        printf("[CHECKPOINT] ERROR: Cannot open %s for writing\n", path.c_str());
        return;
    }

    // Write header
    CheckpointHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "GEC", 4);  // includes null terminator in 4th byte
    hdr.version = 1;
    strncpy(hdr.stage, stage_names[stage], sizeof(hdr.stage) - 1);

    hdr.lloyd_strategy = data.strategy.lloyd_strategy;
    hdr.funcgrad_strategy = data.strategy.funcgrad_strategy;
    hdr.nn_strategy = data.strategy.nn_strategy;
    hdr.adjust_strategy = data.strategy.adjust_strategy;
    hdr.sample_strategy = data.strategy.sample_strategy;
    hdr.extract_strategy = data.strategy.extract_strategy;
    hdr.solve_strategy = data.strategy.solve_strategy;

    hdr.nb_points = data.nb_points;
    hdr.dimension = data.dimension;
    hdr.nb_lloyd_iter = data.nb_lloyd_iter;
    hdr.nb_newton_iter = data.nb_newton_iter;
    hdr.newton_m = data.newton_m;
    hdr.volumetric = data.volumetric ? 1 : 0;
    hdr.nb_vertices = 0;  // filled from mesh on load
    hdr.nb_facets = 0;

    strncpy(hdr.input_mesh, data.input_mesh_path.c_str(),
            sizeof(hdr.input_mesh) - 1);
    hdr.timestamp = (long long)time(nullptr);

    fwrite(&hdr, sizeof(hdr), 1, fp);

    // Write stage index
    int stage_idx = (int)stage;
    fwrite(&stage_idx, sizeof(int), 1, fp);

    // ---- Always save: CVT configuration ----
    geo_ser::Save(fp, data.dimension);
    geo_ser::Save(fp, data.nb_points);
    geo_ser::Save(fp, data.nb_lloyd_iter);
    geo_ser::Save(fp, data.nb_newton_iter);
    geo_ser::Save(fp, data.newton_m);
    geo_ser::Save(fp, (int)data.volumetric);
    geo_ser::Save(fp, (int)data.use_RVC_centroids);
    geo_ser::Save(fp, (int)data.constrained_cvt);
    geo_ser::Save(fp, (int)data.multi_nerve);
    geo_ser::Save(fp, (int)data.adjust);
    geo_ser::Save(fp, data.adjust_max_edge_distance);
    geo_ser::Save(fp, data.adjust_border_importance);

    // Strategy
    geo_ser::Save(fp, data.strategy.lloyd_strategy);
    geo_ser::Save(fp, data.strategy.funcgrad_strategy);
    geo_ser::Save(fp, data.strategy.nn_strategy);
    geo_ser::Save(fp, data.strategy.adjust_strategy);
    geo_ser::Save(fp, data.strategy.sample_strategy);
    geo_ser::Save(fp, data.strategy.extract_strategy);
    geo_ser::Save(fp, data.strategy.solve_strategy);

    // Input mesh path
    geo_ser::SaveString(fp, data.input_mesh_path);

    // ---- Seed points (available after post-sample) ----
    if (stage >= STAGE_POST_SAMPLE) {
        geo_ser::SaveVec(fp, data.points);
    }

    // ---- Locked points ----
    if (stage >= STAGE_POST_SAMPLE) {
        geo_ser::SaveBoolVec(fp, data.point_is_locked);
    }

    // ---- Output mesh (available after post-extract) ----
    if (stage >= STAGE_POST_EXTRACT) {
        geo_ser::Save(fp, data.nb_out_vertices);
        geo_ser::Save(fp, data.nb_out_triangles);
        geo_ser::SaveVec(fp, data.out_vertices);
        geo_ser::SaveVec(fp, data.out_triangles);
    }

    fclose(fp);

    // Print summary
    long file_size = 0;
    struct stat st;
    if (stat(path.c_str(), &st) == 0) file_size = st.st_size;
    printf("[CHECKPOINT] Saved '%s' to %s (%.1f MB)\n",
           stage_names[stage], path.c_str(), file_size / (1024.0 * 1024.0));
    printf("[CHECKPOINT]   strategies: lloyd=%d funcgrad=%d nn=%d\n",
           hdr.lloyd_strategy, hdr.funcgrad_strategy, hdr.nn_strategy);
    printf("[CHECKPOINT]   points=%d dim=%d lloyd_iter=%d newton_iter=%d\n",
           hdr.nb_points, hdr.dimension, hdr.nb_lloyd_iter, hdr.nb_newton_iter);
}

// ============================================================
// Load checkpoint
// ============================================================

PipelineStage load_checkpoint(
    CheckpointData& data,
    const char* dir,
    PipelineStage stage
) {
    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        printf("[CHECKPOINT] ERROR: Cannot open %s for reading\n", path.c_str());
        return STAGE_NONE;
    }

    // Read header
    CheckpointHeader hdr;
    fread(&hdr, sizeof(hdr), 1, fp);
    if (memcmp(hdr.magic, "GEC", 4) != 0) {
        printf("[CHECKPOINT] ERROR: Invalid magic in %s\n", path.c_str());
        fclose(fp);
        return STAGE_NONE;
    }
    if (hdr.version != 1) {
        printf("[CHECKPOINT] ERROR: Unsupported version %d in %s\n",
               hdr.version, path.c_str());
        fclose(fp);
        return STAGE_NONE;
    }

    int stage_idx;
    fread(&stage_idx, sizeof(int), 1, fp);
    PipelineStage saved_stage = (PipelineStage)stage_idx;

    printf("[CHECKPOINT] Loading '%s' from %s\n",
           stage_names[saved_stage], path.c_str());
    printf("[CHECKPOINT]   strategies: lloyd=%d funcgrad=%d nn=%d\n",
           hdr.lloyd_strategy, hdr.funcgrad_strategy, hdr.nn_strategy);
    printf("[CHECKPOINT]   points=%d dim=%d input=%s\n",
           hdr.nb_points, hdr.dimension, hdr.input_mesh);

    // Read CVT configuration
    geo_ser::Read(fp, data.dimension);
    geo_ser::Read(fp, data.nb_points);
    geo_ser::Read(fp, data.nb_lloyd_iter);
    geo_ser::Read(fp, data.nb_newton_iter);
    geo_ser::Read(fp, data.newton_m);

    int tmp;
    geo_ser::Read(fp, tmp); data.volumetric = (tmp != 0);
    geo_ser::Read(fp, tmp); data.use_RVC_centroids = (tmp != 0);
    geo_ser::Read(fp, tmp); data.constrained_cvt = (tmp != 0);
    geo_ser::Read(fp, tmp); data.multi_nerve = (tmp != 0);
    geo_ser::Read(fp, tmp); data.adjust = (tmp != 0);
    geo_ser::Read(fp, data.adjust_max_edge_distance);
    geo_ser::Read(fp, data.adjust_border_importance);

    // Strategy
    geo_ser::Read(fp, data.strategy.lloyd_strategy);
    geo_ser::Read(fp, data.strategy.funcgrad_strategy);
    geo_ser::Read(fp, data.strategy.nn_strategy);
    geo_ser::Read(fp, data.strategy.adjust_strategy);
    geo_ser::Read(fp, data.strategy.sample_strategy);
    geo_ser::Read(fp, data.strategy.extract_strategy);
    geo_ser::Read(fp, data.strategy.solve_strategy);

    // Input mesh path
    geo_ser::ReadString(fp, data.input_mesh_path);

    // Seed points
    if (saved_stage >= STAGE_POST_SAMPLE) {
        geo_ser::ReadVec(fp, data.points);
    }

    // Locked points
    if (saved_stage >= STAGE_POST_SAMPLE) {
        geo_ser::ReadBoolVec(fp, data.point_is_locked);
    }

    // Output mesh
    if (saved_stage >= STAGE_POST_EXTRACT) {
        geo_ser::Read(fp, data.nb_out_vertices);
        geo_ser::Read(fp, data.nb_out_triangles);
        geo_ser::ReadVec(fp, data.out_vertices);
        geo_ser::ReadVec(fp, data.out_triangles);
    }

    fclose(fp);
    return saved_stage;
}

} // namespace geo_ckpt
