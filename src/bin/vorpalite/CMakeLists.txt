set(APP_NAME vorpalite)
aux_source_directories(SOURCES "" .)

# Add checkpoint system sources (compiled here to avoid geogram visibility issues)
list(APPEND SOURCES
    ${PROJECT_SOURCE_DIR}/src/lib/geogram/voronoi/cuda/geo_checkpoint.cpp
)

vor_add_executable(${APP_NAME} ${SOURCES})
target_link_libraries(${APP_NAME} geogram)

# Include path for checkpoint/cuda headers
target_include_directories(${APP_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/lib
    ${PROJECT_SOURCE_DIR}/src/lib/geogram/voronoi/cuda
)

# Link CUDA library if enabled
if(GEOGRAM_WITH_CUDA)
    target_link_libraries(${APP_NAME} geogram_cuda CUDA::cudart)
endif()

install_runtime_targets(${APP_NAME})

set_target_properties(${APP_NAME} PROPERTIES FOLDER "GEOGRAM/Programs")
