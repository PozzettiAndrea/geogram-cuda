/* Vorpaline - geogram demo program
 *
 * Extended with:
 *   - CUDA-accelerated CVT (centroid computation, funcgrad, NN)
 *   - Pipeline checkpoint system (save/resume at stage boundaries)
 *   - Strategy optionality (CPU vs CUDA per component)
 *
 * See -list-stages and -help for usage.
 */

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/progress.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/process.h>
#include <geogram/basic/file_system.h>
#include <geogram/basic/geometry_nd.h>

#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_preprocessing.h>
#include <geogram/mesh/mesh_surface_intersection.h>
#include <geogram/mesh/mesh_reorder.h>
#include <geogram/mesh/mesh_frame_field.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/mesh/mesh_decimate.h>
#include <geogram/mesh/mesh_remesh.h>

#include <geogram/delaunay/LFS.h>
#include <geogram/voronoi/CVT.h>
#include <geogram/voronoi/RVD_mesh_builder.h>
#include <geogram/points/co3ne.h>

#include <geogram/third_party/PoissonRecon/poisson_geogram.h>
#include <geogram/numerics/optimizer.h>
#include <geogram/NL/nl.h>

// Checkpoint + strategy system
#include <geogram/voronoi/cuda/geo_checkpoint.h>

#ifdef WITH_CUDA
#include <geogram/voronoi/cuda/cvt_cuda.h>
#include <cuda_runtime.h>
#endif

#include <typeinfo>
#include <algorithm>
#include <cstring>

namespace {

    using namespace GEO;

    // ================================================================
    // Original helper functions (unchanged)
    // ================================================================

    void remove_small_components(Mesh& M, double min_comp_area) {
        if(min_comp_area == 0.0) {
            return;
        }
        index_t nb_f_removed = M.facets.nb();
        remove_small_connected_components(M, min_comp_area);
        nb_f_removed -= M.facets.nb();
        if(nb_f_removed != 0) {
            double radius = bbox_diagonal(M);
            double epsilon = CmdLine::get_arg_percent(
                "pre:epsilon", radius
            );
            mesh_repair(M, MESH_REPAIR_DEFAULT, epsilon);
        }
    }

    void reconstruct(Mesh& M_in) {
        Logger::div("reconstruction");

        Logger::out("Co3Ne") << "Preparing data" << std::endl;

        index_t Psmooth_iter = CmdLine::get_arg_uint("co3ne:Psmooth_iter");
        index_t nb_neigh = CmdLine::get_arg_uint("co3ne:nb_neighbors");

        if(CmdLine::get_arg("algo:reconstruct") == "Poisson") {
            if(Psmooth_iter != 0) {
                Co3Ne_smooth(M_in, nb_neigh, Psmooth_iter);
            }
            bool has_normals = false;
            {
                Attribute<double> normal;
                normal.bind_if_is_defined(M_in.vertices.attributes(), "normal");
                has_normals = (normal.is_bound() && normal.dimension() == 3);
            }

            if(!has_normals) {
                if(M_in.facets.nb() != 0) {
                    Attribute<double> normal;
                    normal.bind_if_is_defined(
                        M_in.vertices.attributes(),"normal"
                    );
                    if(!normal.is_bound()) {
                        normal.create_vector_attribute(
                            M_in.vertices.attributes(), "normal", 3
                        );
                    }
                    for(index_t i=0; i<M_in.vertices.nb()*3; ++i) {
                        normal[i]=0.0;
                    }
                    for(index_t f=0; f<M_in.facets.nb(); ++f) {
                        vec3 N = Geom::mesh_facet_normal(M_in, f);
                        for(index_t lv=0; lv<M_in.facets.nb_vertices(f); ++lv) {
                            index_t v = M_in.facets.vertex(f,lv);
                            normal[3*v  ] += N.x;
                            normal[3*v+1] += N.y;
                            normal[3*v+2] += N.z;
                        }
                    }
                    for(index_t v=0; v<M_in.vertices.nb(); ++v) {
                        vec3 N(normal[3*v],normal[3*v+1],normal[3*v+2]);
                        N = normalize(N);
                        normal[3*v  ]=N.x;
                        normal[3*v+1]=N.y;
                        normal[3*v+2]=N.z;
                    }
                } else {
                    Logger::out("Poisson")
                        << "Dataset has no normals, estimating them"
                        << std::endl;
                    Logger::out("Poisson")
                        << "(result may be not so good, normals may be incoherent)"
                        << std::endl;
                    Co3Ne_compute_normals(M_in, nb_neigh, true);
                }
            }

            index_t depth = CmdLine::get_arg_uint("poisson:octree_depth");
            Mesh M_out;
            PoissonReconstruction recons;
            recons.set_depth(depth);
            Logger::out("Reconstruct")
                << "Starting Poisson reconstruction..."
                << std::endl;
            recons.reconstruct(&M_in, &M_out);
            Logger::out("Reconstruct")
                << "Poisson reconstruction done."
                << std::endl;
            MeshElementsFlags what = MeshElementsFlags(
                MESH_VERTICES | MESH_FACETS
            );
            M_in.copy(M_out, true, what);
        } else {
            M_in.facets.clear();

            double bbox_diag = bbox_diagonal(M_in);
            double epsilon = CmdLine::get_arg_percent(
                "pre:epsilon", bbox_diag
            );
            mesh_repair(M_in, MESH_REPAIR_COLOCATE, epsilon);

            double radius = CmdLine::get_arg_percent(
                "co3ne:radius", bbox_diag
            );
            Co3Ne_smooth_and_reconstruct(M_in, nb_neigh, Psmooth_iter, radius);
        }
    }

    bool preprocess(Mesh& M_in) {

        Logger::div("preprocessing");
        Stopwatch W("Pre");
        bool pre = CmdLine::get_arg_bool("pre");

        double radius = bbox_diagonal(M_in);
        double area = Geom::mesh_area(M_in, 3);
        double epsilon = CmdLine::get_arg_percent(
            "pre:epsilon", radius
        );
        double max_area = CmdLine::get_arg_percent(
            "pre:max_hole_area", area
        );
        index_t max_edges = CmdLine::get_arg_uint(
            "pre:max_hole_edges"
        );
        bool remove_internal_shells =
            CmdLine::get_arg_bool("pre:remove_internal_shells");

        index_t nb_bins = CmdLine::get_arg_uint("pre:vcluster_bins");

        if(pre && nb_bins != 0) {
            mesh_decimate_vertex_clustering(M_in, nb_bins);
        } else if(pre && CmdLine::get_arg_bool("pre:intersect")) {
            mesh_repair(M_in, MESH_REPAIR_DEFAULT, epsilon);
            if(max_area != 0.0 && max_edges != 0) {
                fill_holes(M_in, max_area, max_edges);
            }
            MeshSurfaceIntersection intersection(M_in);
            intersection.set_verbose(CmdLine::get_arg_bool("sys:verbose"));
            intersection.set_radial_sort(remove_internal_shells);
            intersection.intersect();
            if(remove_internal_shells) {
                intersection.remove_internal_shells();
            }
            mesh_repair(M_in, MESH_REPAIR_DEFAULT, epsilon);
        } else if(pre && CmdLine::get_arg_bool("pre:repair")) {
            MeshRepairMode mode = MESH_REPAIR_DEFAULT;
            mesh_repair(M_in, mode, epsilon);
        }

        if(pre) {
            remove_small_components(
                M_in, CmdLine::get_arg_percent(
                    "pre:min_comp_area", area
                )
            );
        }

        if(pre && !CmdLine::get_arg_bool("pre:intersect")) {
            if(max_area != 0.0 && max_edges != 0) {
                fill_holes(M_in, max_area, max_edges);
            }
        }

        double anisotropy = 0.02 * CmdLine::get_arg_double("remesh:anisotropy");
        if(anisotropy != 0.0) {
            compute_normals(M_in);
            index_t nb_normal_smooth =
                CmdLine::get_arg_uint("pre:Nsmooth_iter");
            if(nb_normal_smooth != 0) {
                Logger::out("Nsmooth") << "Smoothing normals, "
                                       << nb_normal_smooth
                                       << " iteration(s)" << std::endl;
                simple_Laplacian_smooth(M_in, index_t(nb_normal_smooth), true);
            }
            set_anisotropy(M_in, anisotropy);
        }

        if(CmdLine::get_arg_bool("remesh")) {
            index_t nb_removed = M_in.facets.nb();
            remove_small_facets(M_in, 1e-30);
            nb_removed -= M_in.facets.nb();
            if(nb_removed == 0) {
                Logger::out("Validate")
                    << "Mesh does not have 0-area facets (good)" << std::endl;
            } else {
                Logger::out("Validate")
                    << "Removed " << nb_removed
                    << " 0-area facets" << std::endl;
            }
        }

        double margin = CmdLine::get_arg_percent(
            "pre:margin", radius
        );
        if(pre && margin != 0.0) {
            expand_border(M_in, margin);
        }

        if(M_in.facets.nb() == 0) {
            Logger::warn("Preprocessing")
                << "After pre-processing, got an empty mesh"
                << std::endl;
        }

        return true;
    }

    bool postprocess(Mesh& M_out) {
        Logger::div("postprocessing");
        {
            Stopwatch W("Post");
            if(CmdLine::get_arg_bool("post")) {
                double radius = bbox_diagonal(M_out);
                double area = Geom::mesh_area(M_out, 3);
                if(CmdLine::get_arg_bool("post:repair")) {
                    double epsilon = CmdLine::get_arg_percent(
                        "pre:epsilon", radius
                    );
                    mesh_repair(M_out, MESH_REPAIR_DEFAULT, epsilon);
                }
                remove_small_components(
                    M_out, CmdLine::get_arg_percent(
                        "post:min_comp_area", area
                    )
                );
                double max_area = CmdLine::get_arg_percent(
                    "post:max_hole_area", area
                );
                index_t max_edges = CmdLine::get_arg_uint(
                    "post:max_hole_edges"
                );
                if(max_area != 0.0 && max_edges != 0) {
                    fill_holes(M_out, max_area, max_edges);
                }
                double deg3_dist = CmdLine::get_arg_percent(
                    "post:max_deg3_dist", radius
                );
                while(remove_degree3_vertices(M_out, deg3_dist) != 0) {}
                if(CmdLine::get_arg_bool("post:isect")) {
                    mesh_remove_intersections(M_out);
                }
            }
            orient_normals(M_out);
            if(CmdLine::get_arg_bool("post:compute_normals")) {
                Attribute<double> normal;
                normal.bind_if_is_defined(
                    M_out.vertices.attributes(),
                    "normal"
                );
                if(!normal.is_bound()) {
                    normal.create_vector_attribute(
                        M_out.vertices.attributes(),
                        "normal",
                        3
                    );
                }
                for(index_t f=0; f<M_out.facets.nb(); ++f) {
                    vec3 N = Geom::mesh_facet_normal(M_out,f);
                    N = normalize(N);
                    for(index_t lv=0; lv<M_out.facets.nb_vertices(f); ++lv) {
                        index_t v = M_out.facets.vertex(f,lv);
                        normal[3*v  ] = N.x;
                        normal[3*v+1] = N.y;
                        normal[3*v+2] = N.z;
                    }
                }
            }
        }

        Logger::div("result");
        M_out.show_stats("FinalMesh");
        if(M_out.facets.nb() == 0) {
            Logger::warn("Postprocessing")
                << "After post-processing, got an empty mesh"
                << std::endl;
        }

        return true;
    }

    int polyhedral_mesher(
        const std::string& input_filename, std::string output_filename
    ) {
        Mesh M_in;
        Mesh M_out;
        Mesh M_points;

        Logger::div("Polyhedral meshing");

        if(!mesh_load(input_filename, M_in)) {
            return 1;
        }

        if(M_in.cells.nb() == 0) {
            Logger::out("Poly") << "Mesh is not a volume" << std::endl;
            Logger::out("Poly") << "Trying to tetrahedralize" << std::endl;
            if(!mesh_tetrahedralize(M_in)) {
                return 1;
            }
            M_in.cells.compute_borders();
        }

        index_t dim = M_in.vertices.dimension();
        index_t spec_dim = CmdLine::get_arg_uint("poly:embedding_dim");
        if(spec_dim != 0 && spec_dim <= dim) {
            dim = spec_dim;
        }

        CentroidalVoronoiTesselation CVT(&M_in, coord_index_t(dim));
        CVT.set_volumetric(true);

        if(CmdLine::get_arg("poly:points_file") == "") {

            Logger::div("Generate random samples");

            CVT.compute_initial_sampling(
                CmdLine::get_arg_uint("remesh:nb_pts")
            );

            Logger::div("Optimize sampling");

            try {
                index_t nb_iter = CmdLine::get_arg_uint("opt:nb_Lloyd_iter");
                ProgressTask progress("Lloyd", nb_iter);
                CVT.set_progress_logger(&progress);
                CVT.Lloyd_iterations(nb_iter);
            }
            catch(const TaskCanceled&) {
            }

            try {
                index_t nb_iter = CmdLine::get_arg_uint("opt:nb_Newton_iter");
                ProgressTask progress("Newton", nb_iter);
                CVT.set_progress_logger(&progress);
                CVT.Newton_iterations(nb_iter);
            }
            catch(const TaskCanceled&) {
            }

            CVT.set_progress_logger(nullptr);
        } else {
            if(!mesh_load(CmdLine::get_arg("poly:points_file"), M_points)) {
                return 1;
            }
            CVT.delaunay()->set_vertices(
                M_points.vertices.nb(), M_points.vertices.point_ptr(0)
            );
        }

        CVT.RVD()->set_exact_predicates(true);
        {
            BuildRVDMesh callback(M_out);
            std::string simplify = CmdLine::get_arg("poly:simplify");
            if(simplify == "tets_voronoi_boundary") {
                double angle_threshold =
                    CmdLine::get_arg_double("poly:normal_angle_threshold");
                callback.set_simplify_boundary_facets(true, angle_threshold);
            } else if(simplify == "tets_voronoi") {
                callback.set_simplify_voronoi_facets(true);
            } else if(simplify == "tets") {
                callback.set_simplify_internal_tet_facets(true);
            } else if(simplify == "none") {
                callback.set_simplify_internal_tet_facets(false);
            } else {
                Logger::err("Poly")
                    << simplify << " invalid cells simplification mode"
                    << std::endl;
            }
            callback.set_tessellate_non_convex_facets(
                CmdLine::get_arg_bool("poly:tessellate_non_convex_facets")
            );
            callback.set_shrink(CmdLine::get_arg_double("poly:cells_shrink"));
            callback.set_generate_ids(
                CmdLine::get_arg_bool("poly:generate_ids") ||
                FileSystem::extension(output_filename) == "ovm"
            );
            CVT.RVD()->for_each_polyhedron(callback);
        }

        if(
            FileSystem::extension(output_filename) == "mesh" ||
            FileSystem::extension(output_filename) == "meshb"
        ) {
            Logger::warn("Poly")
                << "Specified file format does not handle polygons"
                << " (falling back to .obj)"
                << std::endl;
            output_filename =
                FileSystem::dir_name(output_filename) + "/" +
                FileSystem::base_name(output_filename) + ".obj";
        }

        if(
            CmdLine::get_arg_bool("poly:generate_ids") &&
            FileSystem::extension(output_filename) != "geogram" &&
            FileSystem::extension(output_filename) != "geogram_ascii"
        ) {
            Logger::warn("Poly") << "Speficied file format does not handle ids"
                                 << " (use .geogram or .geogram_ascii instead)"
                                 << std::endl;
        }

        {
            MeshIOFlags flags;
            flags.set_attributes(MESH_ALL_ATTRIBUTES);
            mesh_save(M_out, output_filename, flags);
        }

        return 0;
    }


    int tetrahedral_mesher(
        const std::string& input_filename, const std::string& output_filename
    ) {
        MeshIOFlags flags;
        flags.set_element(MESH_CELLS);

        Mesh M_in;
        if(!mesh_load(input_filename, M_in, flags)) {
            return 1;
        }
        mesh_tetrahedralize(
            M_in,
            CmdLine::get_arg_bool("tet:preprocess"),
            CmdLine::get_arg_bool("tet:refine"),
            CmdLine::get_arg_double("tet:quality")
        );
        M_in.cells.compute_borders();
        if(!mesh_save(M_in, output_filename, flags)) {
            return 1;
        }
        return 0;
    }

    // ================================================================
    // CUDA-aware Lloyd iterations
    //
    // When lloyd_strategy=cuda, uses GPU kernels for centroid
    // computation instead of the exact RVD clipping.
    // Falls through to the standard CPU path when strategy=cpu.
    // ================================================================

#ifdef WITH_CUDA
    void Lloyd_iterations_cuda(
        CentroidalVoronoiTesselation& CVT_obj,
        Mesh& M_in,
        index_t nb_iter,
        index_t nb_points,
        coord_index_t dim
    ) {
        Logger::out("CVT_CUDA") << "Using CUDA Lloyd iterations ("
                                << nb_iter << " iter, "
                                << nb_points << " seeds)" << std::endl;

        // Extract triangle data from mesh for GPU upload
        index_t nv = M_in.vertices.nb();
        index_t nf = M_in.facets.nb();

        std::vector<double> verts(nv * 3);
        for (index_t v = 0; v < nv; v++) {
            const double* p = M_in.vertices.point_ptr(v);
            verts[v*3+0] = p[0];
            verts[v*3+1] = p[1];
            verts[v*3+2] = p[2];
        }

        std::vector<int> tris(nf * 3);
        for (index_t f = 0; f < nf; f++) {
            for (index_t lv = 0; lv < 3; lv++) {
                tris[f*3+lv] = (int)M_in.facets.vertex(f, lv);
            }
        }

        CvtCudaContext* ctx = cvt_cuda_create_context(
            verts.data(), (int)nv, tris.data(), (int)nf
        );
        if (!ctx) {
            Logger::warn("CVT_CUDA") << "Failed to create CUDA context, "
                                     << "falling back to CPU" << std::endl;
            CVT_obj.Lloyd_iterations(nb_iter);
            return;
        }

        // Get current seed points from CVT
        std::vector<double> mg(nb_points * dim, 0.0);
        std::vector<double> m(nb_points, 0.0);

        for (index_t i = 0; i < nb_iter; i++) {
            std::fill(mg.begin(), mg.end(), 0.0);
            std::fill(m.begin(), m.end(), 0.0);

            cvt_cuda_compute_centroids(
                ctx,
                CVT_obj.embedding(0), dim, (int)nb_points,
                mg.data(), m.data()
            );

            // Update points to centroids
            for (index_t j = 0; j < nb_points; j++) {
                if (m[j] > 1e-30 && !CVT_obj.point_is_locked(j)) {
                    double s = 1.0 / m[j];
                    double* pt = CVT_obj.embedding(j);
                    for (index_t c = 0; c < (index_t)dim; c++) {
                        pt[c] = s * mg[j*dim+c];
                    }
                }
            }

            // Rebuild Delaunay for next iteration (needed for convergence check)
            CVT_obj.delaunay()->set_vertices(nb_points, CVT_obj.embedding(0));
        }

        cvt_cuda_destroy_context(ctx);

        Logger::out("CVT_CUDA") << "CUDA Lloyd iterations done" << std::endl;
    }

    // ================================================================
    // CUDA-aware Newton iterations
    //
    // When funcgrad_strategy=cuda, uses GPU kernels for the CVT
    // objective function and gradient evaluation. This is the inner
    // loop of HLBFGS and is called ~10x per Newton iteration.
    //
    // Key advantage over CPU: skips Delaunay rebuild each funcgrad
    // call — the GPU uses direct nearest-seed assignment instead.
    // ================================================================

    // Static state for the CUDA funcgrad callback (HLBFGS uses plain
    // function pointers, so we need static routing like CVT does)
    static CvtCudaContext* s_cuda_ctx = nullptr;
    static CentroidalVoronoiTesselation* s_cuda_cvt = nullptr;
    static coord_index_t s_cuda_dim = 3;

    static void cuda_funcgrad_CB(
        index_t n, double* x, double& f, double* g
    ) {
        index_t nb_points = n / s_cuda_dim;

        // Zero gradient
        memset(g, 0, n * sizeof(double));
        f = 0.0;

        // GPU computes func + grad (no Delaunay rebuild needed!)
        cvt_cuda_compute_funcgrad(
            s_cuda_ctx,
            x, s_cuda_dim, (int)nb_points,
            &f, g
        );

        // Constrain locked points (zero their gradient)
        if (s_cuda_cvt) {
            for (index_t i = 0; i < nb_points; i++) {
                if (s_cuda_cvt->point_is_locked(i)) {
                    for (index_t c = 0; c < (index_t)s_cuda_dim; c++) {
                        g[i * s_cuda_dim + c] = 0.0;
                    }
                }
            }
        }
    }

    static void cuda_newiteration_CB(
        index_t n, const double* x, double f, const double* g, double gnorm
    ) {
        GEO::geo_argused(n);
        GEO::geo_argused(x);
        GEO::geo_argused(g);
        GEO::geo_argused(gnorm);
        // Could log convergence here
    }

    void Newton_iterations_cuda(
        CentroidalVoronoiTesselation& CVT_obj,
        Mesh& M_in,
        index_t nb_iter,
        index_t m,
        index_t nb_points,
        coord_index_t dim
    ) {
        Logger::out("CVT_CUDA") << "Using CUDA Newton iterations ("
                                << nb_iter << " iter, m=" << m
                                << ", " << nb_points << " seeds)" << std::endl;

        // Extract mesh for GPU
        index_t nv = M_in.vertices.nb();
        index_t nf = M_in.facets.nb();

        std::vector<double> verts(nv * 3);
        for (index_t v = 0; v < nv; v++) {
            const double* p = M_in.vertices.point_ptr(v);
            verts[v*3+0] = p[0];
            verts[v*3+1] = p[1];
            verts[v*3+2] = p[2];
        }

        std::vector<int> tris(nf * 3);
        for (index_t f = 0; f < nf; f++) {
            for (index_t lv = 0; lv < 3; lv++) {
                tris[f*3+lv] = (int)M_in.facets.vertex(f, lv);
            }
        }

        CvtCudaContext* ctx = cvt_cuda_create_context(
            verts.data(), (int)nv, tris.data(), (int)nf
        );
        if (!ctx) {
            Logger::warn("CVT_CUDA") << "Failed to create CUDA context, "
                                     << "falling back to CPU" << std::endl;
            CVT_obj.Newton_iterations(nb_iter, m);
            return;
        }

        // Set up static state for callback
        s_cuda_ctx = ctx;
        s_cuda_cvt = &CVT_obj;
        s_cuda_dim = dim;

        // Temporarily take over CVT's singleton to prevent conflict
        CVT_obj.done_current();

        // Create HLBFGS optimizer and drive it with our CUDA callback
        Optimizer_var optimizer = Optimizer::create("HLBFGS");
        if (optimizer.is_null()) {
            Logger::warn("CVT_CUDA") << "HLBFGS not available, falling back to CPU"
                                     << std::endl;
            CVT_obj.make_current();
            cvt_cuda_destroy_context(ctx);
            s_cuda_ctx = nullptr;
            s_cuda_cvt = nullptr;
            CVT_obj.Newton_iterations(nb_iter, m);
            return;
        }

        index_t n = nb_points * dim;

        optimizer->set_epsg(0.0);
        optimizer->set_epsf(0.0);
        optimizer->set_epsx(0.0);
        optimizer->set_newiteration_callback(cuda_newiteration_CB);
        optimizer->set_funcgrad_callback(cuda_funcgrad_CB);
        optimizer->set_N(n);
        optimizer->set_M(m);
        optimizer->set_max_iter(nb_iter);
        optimizer->optimize(CVT_obj.embedding(0));

        // Restore CVT singleton
        CVT_obj.make_current();

        // Rebuild Delaunay with final positions (needed for extract stage)
        CVT_obj.delaunay()->set_vertices(nb_points, CVT_obj.embedding(0));

        // Cleanup
        cvt_cuda_destroy_context(ctx);
        s_cuda_ctx = nullptr;
        s_cuda_cvt = nullptr;

        Logger::out("CVT_CUDA") << "CUDA Newton iterations done" << std::endl;
    }
#endif

    // ================================================================
    // CUDA-accelerated mesh_adjust_surface
    //
    // Replaces the CPU AABB ray-cast loop with GPU brute-force
    // ray-triangle intersection. The linear solve (OpenNL) stays on CPU.
    // ================================================================

#ifdef WITH_CUDA
    void mesh_adjust_surface_cuda(
        Mesh& surface, Mesh& reference,
        double max_edge_distance, double border_importance,
        int solve_strategy = 0
    ) {
        Logger::out("CVT_CUDA") << "Using CUDA surface adjustment" << std::endl;

        index_t nb_sv = surface.vertices.nb();
        index_t nb_sf = surface.facets.nb();
        index_t nb_rv = reference.vertices.nb();
        index_t nb_rf = reference.facets.nb();

        // Extract reference mesh for GPU
        std::vector<double> ref_verts(nb_rv * 3);
        for (index_t v = 0; v < nb_rv; v++) {
            const double* p = reference.vertices.point_ptr(v);
            ref_verts[v*3+0] = p[0]; ref_verts[v*3+1] = p[1]; ref_verts[v*3+2] = p[2];
        }
        std::vector<int> ref_tris(nb_rf * 3);
        for (index_t f = 0; f < nb_rf; f++) {
            for (index_t lv = 0; lv < 3; lv++) {
                ref_tris[f*3+lv] = (int)reference.facets.vertex(f, lv);
            }
        }

        // Compute vertex normals and average edge lengths on surface
        std::vector<vec3> Nv(nb_sv, vec3(0.0, 0.0, 0.0));
        std::vector<double> Lv(nb_sv, 0.0);
        std::vector<index_t> Cv(nb_sv, 0);

        for (index_t f: surface.facets) {
            vec3 n = Geom::mesh_facet_normal(surface, f);
            index_t d = surface.facets.nb_vertices(f);
            for (index_t lv = 0; lv < d; ++lv) {
                index_t v1 = surface.facets.vertex(f, lv);
                index_t v2 = surface.facets.vertex(f, (lv==d-1)?0:lv+1);
                Nv[v1] += n;
                double l = length(surface.vertices.point(v1) - surface.vertices.point(v2));
                Lv[v1] += l; Lv[v2] += l;
                Cv[v1]++; Cv[v2]++;
            }
        }
        for (index_t v: surface.vertices) {
            if (Cv[v] != 0) Lv[v] /= double(Cv[v]);
        }

        // Build query arrays for GPU ray-cast
        std::vector<double> qp(nb_sv * 3);
        std::vector<double> qd(nb_sv * 3);
        std::vector<double> md(nb_sv);

        for (index_t v = 0; v < nb_sv; v++) {
            const vec3& p = surface.vertices.point(v);
            qp[v*3+0] = p.x; qp[v*3+1] = p.y; qp[v*3+2] = p.z;
            qd[v*3+0] = Nv[v].x; qd[v*3+1] = Nv[v].y; qd[v*3+2] = Nv[v].z;
            md[v] = max_edge_distance * Lv[v];
        }

        // GPU ray-cast: find nearest point for each vertex
        std::vector<double> nearest(nb_sv * 3);
        std::vector<int> hit(nb_sv);
        cvt_cuda_ray_mesh_nearest(
            ref_verts.data(), ref_tris.data(),
            (int)nb_rv, (int)nb_rf,
            qp.data(), qd.data(), md.data(), (int)nb_sv,
            nearest.data(), hit.data()
        );

        // Also do facet center queries
        std::vector<double> fqp(nb_sf * 3);
        std::vector<double> fqd(nb_sf * 3);
        std::vector<double> fmd(nb_sf);
        std::vector<double> fnearest(nb_sf * 3);
        std::vector<int> fhit(nb_sf);

        for (index_t f: surface.facets) {
            index_t d = surface.facets.nb_vertices(f);
            vec3 Pf(0,0,0), Nf(0,0,0);
            double Lf = 0;
            for (index_t lv = 0; lv < d; ++lv) {
                index_t v = surface.facets.vertex(f, lv);
                Pf += surface.vertices.point(v);
                Nf += Nv[v];
                Lf += Lv[v];
            }
            Pf = (1.0/double(d)) * Pf;
            Lf = (1.0/double(d)) * Lf;
            fqp[f*3+0] = Pf.x; fqp[f*3+1] = Pf.y; fqp[f*3+2] = Pf.z;
            fqd[f*3+0] = Nf.x; fqd[f*3+1] = Nf.y; fqd[f*3+2] = Nf.z;
            fmd[f] = max_edge_distance * Lf;
        }

        cvt_cuda_ray_mesh_nearest(
            ref_verts.data(), ref_tris.data(),
            (int)nb_rv, (int)nb_rf,
            fqp.data(), fqd.data(), fmd.data(), (int)nb_sf,
            fnearest.data(), fhit.data()
        );

        // Build and solve least-squares system (same as CPU path)
        nlNewContext();
        if (solve_strategy == 1) {
            if (nlInitExtension("CUDA")) {
                Logger::out("CVT_CUDA") << "OpenNL CUDA solver enabled" << std::endl;
            } else {
                Logger::warn("CVT_CUDA") << "OpenNL CUDA not available, using CPU solver" << std::endl;
            }
        }
        nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
        nlSolverParameteri(NL_NB_VARIABLES, NLint(nb_sv));
        nlBegin(NL_SYSTEM);
        nlBegin(NL_MATRIX);

        // Per-vertex constraints
        for (index_t v = 0; v < nb_sv; v++) {
            vec3 Qv(nearest[v*3+0], nearest[v*3+1], nearest[v*3+2]);
            for (index_t c = 0; c < 3; ++c) {
                nlBegin(NL_ROW);
                nlCoefficient(v, Nv[v][c]);
                nlRightHandSide(Qv[c] - surface.vertices.point_ptr(v)[c]);
                nlEnd(NL_ROW);
            }
        }

        // Per-facet constraints
        for (index_t f: surface.facets) {
            index_t d = surface.facets.nb_vertices(f);
            vec3 Pf(0,0,0);
            for (index_t lv = 0; lv < d; ++lv) {
                Pf += surface.vertices.point(surface.facets.vertex(f, lv));
            }
            Pf = (1.0/double(d)) * Pf;
            vec3 Qf(fnearest[f*3+0], fnearest[f*3+1], fnearest[f*3+2]);
            for (index_t c = 0; c < 3; ++c) {
                nlBegin(NL_ROW);
                for (index_t lv = 0; lv < d; ++lv) {
                    index_t v = surface.facets.vertex(f, lv);
                    nlCoefficient(v, Nv[v][c] / double(d));
                }
                nlRightHandSide(Qf[c] - Pf[c]);
                nlEnd(NL_ROW);
            }
        }

        nlEnd(NL_MATRIX);
        nlEnd(NL_SYSTEM);
        nlSolve();

        for (index_t v: surface.vertices) {
            vec3& p = surface.vertices.point(v);
            p += nlGetVariable(v) * Nv[v];
        }

        nlDeleteContext(nlGetCurrent());
        Logger::out("CVT_CUDA") << "CUDA surface adjustment done" << std::endl;
    }
#endif

    // ================================================================
    // Staged remeshing pipeline with checkpoint support
    // ================================================================

    int remesh_staged(
        const std::string& input_filename,
        const std::string& output_filename,
        Mesh& M_in, Mesh& M_out,
        const geo_ckpt::StrategyConfig& strategy,
        const std::string& save_dir,
        geo_ckpt::PipelineStage save_at,
        geo_ckpt::PipelineStage run_from,
        geo_ckpt::PipelineStage run_to,
        bool save_all
    ) {
        using namespace geo_ckpt;

        // ---- Pipeline helpers ----
        auto should_run = [&](PipelineStage stage) -> bool {
            if (run_from != STAGE_NONE && stage <= run_from) return false;
            if (run_to != STAGE_NONE && stage > run_to) return false;
            return true;
        };

        auto maybe_save = [&](PipelineStage stage, CheckpointData& data) {
            if (!save_dir.empty() && (save_all || stage == save_at)) {
                save_checkpoint(data, stage, save_dir.c_str());
            }
        };

        auto should_stop = [&](PipelineStage stage) -> bool {
            return (run_to != STAGE_NONE && stage >= run_to);
        };

        // ---- Checkpoint data bundle ----
        CheckpointData ckpt;
        ckpt.input_mesh_path = input_filename;
        ckpt.strategy = strategy;

        index_t nb_points = CmdLine::get_arg_uint("remesh:nb_pts");
        index_t nb_Lloyd_iter = CmdLine::get_arg_uint("opt:nb_Lloyd_iter");
        index_t nb_Newton_iter = CmdLine::get_arg_uint("opt:nb_Newton_iter");
        index_t Newton_m = CmdLine::get_arg_uint("opt:Newton_m");
        coord_index_t dim = 3;
        double anisotropy = 0.02 * CmdLine::get_arg_double("remesh:anisotropy");
        if (anisotropy != 0.0 && M_in.vertices.dimension() >= 6) {
            dim = 6;
        }

        ckpt.dimension = dim;
        ckpt.nb_points = nb_points;
        ckpt.nb_lloyd_iter = nb_Lloyd_iter;
        ckpt.nb_newton_iter = nb_Newton_iter;
        ckpt.newton_m = Newton_m;
        ckpt.use_RVC_centroids = CmdLine::get_arg_bool("remesh:RVC_centroids");
        ckpt.multi_nerve = CmdLine::get_arg_bool("remesh:multi_nerve");

        // ---- Load checkpoint if resuming ----
        if (run_from != STAGE_NONE) {
            if (save_dir.empty()) {
                Logger::err("Pipeline") << "-run-from requires -save-dir" << std::endl;
                return 1;
            }
            PipelineStage loaded = load_checkpoint(ckpt, save_dir.c_str(), run_from);
            if (loaded == STAGE_NONE) {
                Logger::err("Pipeline") << "Failed to load checkpoint for stage '"
                                        << stage_name(run_from) << "'" << std::endl;
                return 1;
            }
            Logger::out("Pipeline") << "Resuming after stage '"
                                    << stage_name(run_from) << "'" << std::endl;
            nb_points = ckpt.nb_points;
            nb_Lloyd_iter = ckpt.nb_lloyd_iter;
            nb_Newton_iter = ckpt.nb_newton_iter;
            Newton_m = ckpt.newton_m;
            dim = (coord_index_t)ckpt.dimension;
        }

        // ---- Create CVT ----
        CentroidalVoronoiTesselation CVT(&M_in, dim);

        // ================================================================
        // STAGE: POST-LOAD
        // ================================================================
        if (should_run(STAGE_POST_LOAD)) {
            Logger::div("Stage: post-load");
            // Preprocessing already done in main()
            double gradation = CmdLine::get_arg_double("remesh:gradation");
            if (gradation != 0.0) {
                compute_sizing_field(
                    M_in, gradation, CmdLine::get_arg_uint("remesh:lfs_samples")
                );
            }
            maybe_save(STAGE_POST_LOAD, ckpt);
            if (should_stop(STAGE_POST_LOAD)) return 0;
        }

        // ================================================================
        // STAGE: POST-SAMPLE
        // ================================================================
        if (should_run(STAGE_POST_SAMPLE)) {
            Logger::div("Stage: post-sample");
            Stopwatch W("Sample");

            if (run_from >= STAGE_POST_SAMPLE && !ckpt.points.empty()) {
                // Restore points from checkpoint
                CVT.set_points(nb_points, ckpt.points.data());
                Logger::out("Pipeline") << "Restored " << nb_points
                                        << " seed points from checkpoint" << std::endl;
            } else {
                if (nb_points == 0) {
                    nb_points = M_in.vertices.nb();
                }
#ifdef WITH_CUDA
                if (strategy.sample_strategy == 1) {
                    Logger::out("CVT_CUDA") << "Using CUDA surface sampling ("
                                            << nb_points << " samples)" << std::endl;
                    // Extract mesh data for GPU
                    index_t nv = M_in.vertices.nb();
                    index_t nf = M_in.facets.nb();
                    std::vector<double> verts(nv * 3);
                    for (index_t v = 0; v < nv; v++) {
                        const double* p = M_in.vertices.point_ptr(v);
                        verts[v*3+0] = p[0]; verts[v*3+1] = p[1]; verts[v*3+2] = p[2];
                    }
                    std::vector<int> tris(nf * 3);
                    for (index_t f = 0; f < nf; f++) {
                        for (index_t lv = 0; lv < 3; lv++)
                            tris[f*3+lv] = (int)M_in.facets.vertex(f, lv);
                    }
                    // Compute triangle areas on CPU (simple, small cost)
                    std::vector<double> areas(nf);
                    for (index_t f = 0; f < nf; f++) {
                        int v0 = tris[f*3+0], v1 = tris[f*3+1], v2 = tris[f*3+2];
                        double e1x = verts[v1*3]-verts[v0*3], e1y = verts[v1*3+1]-verts[v0*3+1], e1z = verts[v1*3+2]-verts[v0*3+2];
                        double e2x = verts[v2*3]-verts[v0*3], e2y = verts[v2*3+1]-verts[v0*3+1], e2z = verts[v2*3+2]-verts[v0*3+2];
                        double cx = e1y*e2z - e1z*e2y, cy = e1z*e2x - e1x*e2z, cz = e1x*e2y - e1y*e2x;
                        areas[f] = 0.5 * sqrt(cx*cx + cy*cy + cz*cz);
                    }
                    // Generate samples on GPU
                    std::vector<double> samples(nb_points * 3);
                    cvt_cuda_generate_surface_samples(
                        verts.data(), (int)nv,
                        tris.data(), (int)nf,
                        areas.data(),
                        (int)nb_points, 42ULL,
                        samples.data()
                    );
                    // Note: GPU sampling produces 3D points only.
                    // For dim=6 (anisotropic), we'd need to project to 6D.
                    // For now, set the first 3 coords and zero the rest.
                    if (dim == 3) {
                        CVT.set_points(nb_points, samples.data());
                    } else {
                        // For dim > 3, fall back to CPU sampling
                        Logger::warn("CVT_CUDA") << "GPU sampling only supports dim=3, "
                                                  << "falling back to CPU" << std::endl;
                        CVT.compute_initial_sampling(nb_points, true);
                    }
                } else
#endif
                {
                    CVT.compute_initial_sampling(nb_points, true);
                }
            }

            // Save points to checkpoint data
            ckpt.nb_points = CVT.nb_points();
            ckpt.points.resize(CVT.nb_points() * dim);
            for (index_t i = 0; i < CVT.nb_points(); i++) {
                const double* p = CVT.embedding(i);
                for (index_t c = 0; c < dim; c++) {
                    ckpt.points[i*dim+c] = p[c];
                }
            }

            maybe_save(STAGE_POST_SAMPLE, ckpt);
            if (should_stop(STAGE_POST_SAMPLE)) return 0;
        }

        // Restore points if we're resuming past sampling
        if (run_from >= STAGE_POST_SAMPLE && !ckpt.points.empty()) {
            CVT.set_points(ckpt.nb_points, ckpt.points.data());
        }

        // ================================================================
        // STAGE: POST-LLOYD
        // ================================================================
        if (should_run(STAGE_POST_LLOYD)) {
            Logger::div("Stage: post-lloyd");
            Stopwatch W("Lloyd");

            try {
                ProgressTask progress("Lloyd", nb_Lloyd_iter);
                CVT.set_progress_logger(&progress);

#ifdef WITH_CUDA
                if (strategy.lloyd_strategy == 1) {
                    Lloyd_iterations_cuda(CVT, M_in, nb_Lloyd_iter,
                                         CVT.nb_points(), dim);
                } else {
                    CVT.Lloyd_iterations(nb_Lloyd_iter);
                }
#else
                CVT.Lloyd_iterations(nb_Lloyd_iter);
#endif
            }
            catch(const TaskCanceled&) {
            }
            CVT.set_progress_logger(nullptr);

            // Update checkpoint data with new points
            ckpt.nb_points = CVT.nb_points();
            ckpt.points.resize(CVT.nb_points() * dim);
            for (index_t i = 0; i < CVT.nb_points(); i++) {
                const double* p = CVT.embedding(i);
                for (index_t c = 0; c < dim; c++) {
                    ckpt.points[i*dim+c] = p[c];
                }
            }

            maybe_save(STAGE_POST_LLOYD, ckpt);
            if (should_stop(STAGE_POST_LLOYD)) return 0;
        }

        // ================================================================
        // STAGE: POST-NEWTON
        // ================================================================
        if (should_run(STAGE_POST_NEWTON)) {
            Logger::div("Stage: post-newton");
            Stopwatch W("Newton");

            if (nb_Newton_iter != 0) {
#ifdef WITH_CUDA
                if (strategy.funcgrad_strategy == 1) {
                    Newton_iterations_cuda(
                        CVT, M_in, nb_Newton_iter, Newton_m,
                        CVT.nb_points(), dim
                    );
                } else
#endif
                {
                    try {
                        ProgressTask progress("Newton", nb_Newton_iter);
                        CVT.set_progress_logger(&progress);
                        CVT.Newton_iterations(nb_Newton_iter, Newton_m);
                    }
                    catch(const TaskCanceled&) {
                    }
                    CVT.set_progress_logger(nullptr);
                }
            }

            // Update checkpoint data
            ckpt.nb_points = CVT.nb_points();
            ckpt.points.resize(CVT.nb_points() * dim);
            for (index_t i = 0; i < CVT.nb_points(); i++) {
                const double* p = CVT.embedding(i);
                for (index_t c = 0; c < dim; c++) {
                    ckpt.points[i*dim+c] = p[c];
                }
            }

            maybe_save(STAGE_POST_NEWTON, ckpt);
            if (should_stop(STAGE_POST_NEWTON)) return 0;
        }

        // ================================================================
        // STAGE: POST-EXTRACT
        // ================================================================
        if (should_run(STAGE_POST_EXTRACT)) {
            Logger::div("Stage: post-extract");
            Stopwatch W("Extract");

            CVT.RVD()->delete_threads();

            CVT.set_use_RVC_centroids(ckpt.use_RVC_centroids);
            bool multi_nerve = ckpt.multi_nerve;

            Logger::out("Remesh") << "Computing RVD..." << std::endl;
            CVT.compute_surface(&M_out, multi_nerve);

            if (CmdLine::get_arg_bool("dbg:save_ANN_histo")) {
                Logger::out("ANN")
                    << "Saving histogram to ANN_histo.dat" << std::endl;
                std::ofstream out("ANN_histo.dat");
                CVT.delaunay()->save_histogram(out);
            }

            maybe_save(STAGE_POST_EXTRACT, ckpt);
            if (should_stop(STAGE_POST_EXTRACT)) return 0;
        }

        // ================================================================
        // STAGE: POST-ADJUST
        // ================================================================
        if (should_run(STAGE_POST_ADJUST)) {
            Logger::div("Stage: post-adjust");
            Stopwatch W("Adjust");

            if (ckpt.adjust) {
#ifdef WITH_CUDA
                if (strategy.adjust_strategy == 1) {
                    mesh_adjust_surface_cuda(
                        M_out, M_in,
                        ckpt.adjust_max_edge_distance,
                        ckpt.adjust_border_importance,
                        strategy.solve_strategy
                    );
                } else
#endif
                {
                    mesh_adjust_surface(
                        M_out, M_in,
                        ckpt.adjust_max_edge_distance,
                        false,
                        ckpt.adjust_border_importance
                    );
                }
            }

            maybe_save(STAGE_POST_ADJUST, ckpt);
        }

        return 0;
    }

    // ================================================================
    // Parse strategy flags from command-line (custom args beyond geogram's)
    // ================================================================

    struct ExtraArgs {
        std::string save_dir;
        geo_ckpt::PipelineStage save_at;
        geo_ckpt::PipelineStage run_from;
        geo_ckpt::PipelineStage run_to;
        bool save_all;
        bool list_stages;
        geo_ckpt::StrategyConfig strategy;

        ExtraArgs()
            : save_at(geo_ckpt::STAGE_NONE),
              run_from(geo_ckpt::STAGE_NONE),
              run_to(geo_ckpt::STAGE_NONE),
              save_all(false),
              list_stages(false) {}
    };

    // Parse our custom flags and build a filtered argv for geogram.
    // Returns false on error.
    bool parse_extra_args(int argc, char** argv, ExtraArgs& args,
                          std::vector<char*>& filtered_argv) {
        for (int i = 0; i < argc; i++) {
            bool consumed = false;
            if (i > 0) {  // skip argv[0]
                if (strcmp(argv[i], "-save-dir") == 0 && i + 1 < argc) {
                    args.save_dir = argv[++i]; consumed = true;
                } else if (strcmp(argv[i], "-save-at") == 0 && i + 1 < argc) {
                    args.save_at = geo_ckpt::stage_from_name(argv[++i]);
                    if (args.save_at == geo_ckpt::STAGE_NONE) {
                        fprintf(stderr, "Unknown stage: %s\n", argv[i]);
                        return false;
                    }
                    consumed = true;
                } else if (strcmp(argv[i], "-save-all") == 0) {
                    args.save_all = true; consumed = true;
                } else if (strcmp(argv[i], "-run-from") == 0 && i + 1 < argc) {
                    args.run_from = geo_ckpt::stage_from_name(argv[++i]);
                    if (args.run_from == geo_ckpt::STAGE_NONE) {
                        fprintf(stderr, "Unknown stage: %s\n", argv[i]);
                        return false;
                    }
                    consumed = true;
                } else if (strcmp(argv[i], "-run-to") == 0 && i + 1 < argc) {
                    args.run_to = geo_ckpt::stage_from_name(argv[++i]);
                    if (args.run_to == geo_ckpt::STAGE_NONE) {
                        fprintf(stderr, "Unknown stage: %s\n", argv[i]);
                        return false;
                    }
                    consumed = true;
                } else if (strcmp(argv[i], "-list-stages") == 0) {
                    args.list_stages = true; consumed = true;
                } else if (strcmp(argv[i], "-lloyd-strategy") == 0 && i + 1 < argc) {
                    const char* s = argv[++i];
                    if (strcmp(s, "cpu") == 0)       args.strategy.lloyd_strategy = 0;
                    else if (strcmp(s, "cuda") == 0) args.strategy.lloyd_strategy = 1;
                    else { fprintf(stderr, "Unknown -lloyd-strategy: %s\n", s); return false; }
                    consumed = true;
                } else if (strcmp(argv[i], "-funcgrad-strategy") == 0 && i + 1 < argc) {
                    const char* s = argv[++i];
                    if (strcmp(s, "cpu") == 0)       args.strategy.funcgrad_strategy = 0;
                    else if (strcmp(s, "cuda") == 0) args.strategy.funcgrad_strategy = 1;
                    else { fprintf(stderr, "Unknown -funcgrad-strategy: %s\n", s); return false; }
                    consumed = true;
                } else if (strcmp(argv[i], "-nn-strategy") == 0 && i + 1 < argc) {
                    const char* s = argv[++i];
                    if (strcmp(s, "cpu") == 0)       args.strategy.nn_strategy = 0;
                    else if (strcmp(s, "cuda") == 0) args.strategy.nn_strategy = 1;
                    else { fprintf(stderr, "Unknown -nn-strategy: %s\n", s); return false; }
                    consumed = true;
                } else if (strcmp(argv[i], "-adjust-strategy") == 0 && i + 1 < argc) {
                    const char* s = argv[++i];
                    if (strcmp(s, "cpu") == 0)       args.strategy.adjust_strategy = 0;
                    else if (strcmp(s, "cuda") == 0) args.strategy.adjust_strategy = 1;
                    else { fprintf(stderr, "Unknown -adjust-strategy: %s\n", s); return false; }
                    consumed = true;
                } else if (strcmp(argv[i], "-sample-strategy") == 0 && i + 1 < argc) {
                    const char* s = argv[++i];
                    if (strcmp(s, "cpu") == 0)       args.strategy.sample_strategy = 0;
                    else if (strcmp(s, "cuda") == 0) args.strategy.sample_strategy = 1;
                    else { fprintf(stderr, "Unknown -sample-strategy: %s\n", s); return false; }
                    consumed = true;
                } else if (strcmp(argv[i], "-extract-strategy") == 0 && i + 1 < argc) {
                    const char* s = argv[++i];
                    if (strcmp(s, "cpu") == 0)       args.strategy.extract_strategy = 0;
                    else if (strcmp(s, "cuda") == 0) args.strategy.extract_strategy = 1;
                    else { fprintf(stderr, "Unknown -extract-strategy: %s\n", s); return false; }
                    consumed = true;
                } else if (strcmp(argv[i], "-solve-strategy") == 0 && i + 1 < argc) {
                    const char* s = argv[++i];
                    if (strcmp(s, "cpu") == 0)       args.strategy.solve_strategy = 0;
                    else if (strcmp(s, "cuda") == 0) args.strategy.solve_strategy = 1;
                    else { fprintf(stderr, "Unknown -solve-strategy: %s\n", s); return false; }
                    consumed = true;
                } else if (strcmp(argv[i], "-all-cuda") == 0) {
                    args.strategy.lloyd_strategy = 1;
                    args.strategy.funcgrad_strategy = 1;
                    args.strategy.nn_strategy = 1;
                    args.strategy.adjust_strategy = 1;
                    args.strategy.sample_strategy = 1;
                    args.strategy.extract_strategy = 1;
                    args.strategy.solve_strategy = 1;
                    consumed = true;
                } else if (strcmp(argv[i], "-cuda-safe") == 0) {
                    args.strategy.lloyd_strategy = 1;
                    args.strategy.funcgrad_strategy = 1;
                    args.strategy.sample_strategy = 1;
                    args.strategy.solve_strategy = 1;
                    consumed = true;
                }
            }
            if (!consumed) {
                filtered_argv.push_back(argv[i]);
            }
        }
        return true;
    }

}

int main(int argc, char** argv) {
    using namespace GEO;

#ifdef WITH_CUDA
    cudaFree(0);  // Warm up GPU
#endif

    GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);

    try {

        Stopwatch total("Total time");

        CmdLine::import_arg_group("standard");
        CmdLine::import_arg_group("pre");
        CmdLine::import_arg_group("remesh");
        CmdLine::import_arg_group("algo");
        CmdLine::import_arg_group("post");
        CmdLine::import_arg_group("opt");
        CmdLine::import_arg_group("co3ne");
        CmdLine::import_arg_group("tet");
        CmdLine::import_arg_group("poly");

        // Parse our extra args and build filtered argv for geogram
        ExtraArgs extra;
        std::vector<char*> filtered_argv;
        if (!parse_extra_args(argc, argv, extra, filtered_argv)) {
            return 1;
        }
        int filtered_argc = (int)filtered_argv.size();

        if (extra.list_stages) {
            geo_ckpt::list_stages();
            printf("\nStrategy options:\n");
            printf("  -lloyd-strategy cpu|cuda       Lloyd centroid computation\n");
            printf("  -funcgrad-strategy cpu|cuda    CVT function+gradient\n");
            printf("  -nn-strategy cpu|cuda          Nearest-neighbor queries\n");
            printf("  -adjust-strategy cpu|cuda      Surface adjustment ray-casting\n");
            printf("  -sample-strategy cpu|cuda      Initial sampling\n");
            printf("  -extract-strategy cpu|cuda     RDT extraction (approx on cuda)\n");
            printf("  -solve-strategy cpu|cuda       Adjust linear solve (OpenNL)\n");
            printf("  -all-cuda                      Enable CUDA for all stages\n");
            printf("  -cuda-safe                     CUDA for safe stages only\n");
            printf("\nCheckpoint options:\n");
            printf("  -save-dir <dir>                Directory for .gec files\n");
            printf("  -save-at <stage>               Save after this stage\n");
            printf("  -save-all                      Save after every stage\n");
            printf("  -run-from <stage>              Resume from checkpoint\n");
            printf("  -run-to <stage>                Stop after this stage\n");
            return 0;
        }

        std::vector<std::string> filenames;

        if(!CmdLine::parse(filtered_argc, filtered_argv.data(), filenames, "inputfile <outputfile>")) {
            return 1;
        }

        std::string input_filename = filenames[0];
        std::string output_filename =
            filenames.size() >= 2 ? filenames[1] : std::string("out.meshb");
        Logger::out("I/O") << "Output = " << output_filename << std::endl;
        CmdLine::set_arg("input", input_filename);
        CmdLine::set_arg("output", output_filename);

        // Log strategy configuration
        Logger::out("Strategy") << "lloyd=" << extra.strategy.lloyd_strategy
                                << " funcgrad=" << extra.strategy.funcgrad_strategy
                                << " sample=" << extra.strategy.sample_strategy
                                << " extract=" << extra.strategy.extract_strategy
                                << " adjust=" << extra.strategy.adjust_strategy
                                << " solve=" << extra.strategy.solve_strategy
                                << " (0=cpu, 1=cuda)" << std::endl;

        if(CmdLine::get_arg_bool("tet")) {
            return tetrahedral_mesher(input_filename, output_filename);
        }

        if(CmdLine::get_arg_bool("poly")) {
            return polyhedral_mesher(input_filename, output_filename);
        }

        Mesh M_in, M_out;
        {
            Stopwatch W("Load");
            if(!mesh_load(input_filename, M_in)) {
                return 1;
            }
        }

        if(CmdLine::get_arg_bool("co3ne")) {
            reconstruct(M_in);
        }

        if(!preprocess(M_in)) {
            return 1;
        }

        if(!CmdLine::get_arg_bool("remesh")) {
            if(!postprocess(M_in)) {
                return 1;
            }
            if(!mesh_save(M_in, output_filename)) {
                return 1;
            }
            return 0;
        }

        // ---- Use staged pipeline with checkpoint support ----
        Logger::div("remeshing (staged pipeline)");
        int result = remesh_staged(
            input_filename, output_filename,
            M_in, M_out,
            extra.strategy,
            extra.save_dir,
            extra.save_at,
            extra.run_from,
            extra.run_to,
            extra.save_all
        );

        if (result != 0) return result;

        if(M_out.facets.nb() == 0) {
            Logger::err("Remesh") << "After remesh, got an empty mesh"
                                  << std::endl;
            return 1;
        }

        if(!postprocess(M_out)) {
            return 1;
        }

        if(!mesh_save(M_out, output_filename)) {
            return 1;
        }

    }
    catch(const std::exception& e) {
        std::cerr << "Received an exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
