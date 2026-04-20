cmake_minimum_required(VERSION 3.21)
project(BlackHoleSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GLEW     REQUIRED)
find_package(glfw3    REQUIRED)
find_package(glm      REQUIRED)
find_package(OpenGL   REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(DEPS glfw GLEW::GLEW glm::glm OpenGL::GL)


# ── executável principal CUDA + OpenGL ────────────────────────────────────────
#
# Regra: arquivos .cu são compilados pelo NVCC (device + host code).
#        arquivos .cpp são compilados pelo GCC normalmente.
#        O linker (NVCC) une tudo no final.
#
# gl_engine.cpp  → inclui GLEW/GLFW/GLM → compilado por GCC, sem problema
# geodesic.cu    → inclui só cuda_runtime → compilado por NVCC, sem GLEW/GLFW
# main.cpp       → inclui gl_engine.h    → compilado por GCC

add_executable(BlackHoleCUDA
    main.cpp
    src/engine.cpp
    src/host.cpp   # host code — GCC
    cuda/geodesic.cu         # device code — NVCC
    src/lodepng.cpp
    src/starmap.cpp
    src/perlin.cpp
)

target_link_libraries(BlackHoleCUDA PRIVATE
    ${DEPS}
    CUDA::cudart
)

set_target_properties(BlackHoleCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         75
    LINKER_LANGUAGE            CUDA     # deixa o NVCC fazer o link final
)

get_target_property(CUDA_INCLUDE_DIRS CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(BlackHoleCUDA SYSTEM PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CUDA_INCLUDE_DIRS}>
)

file(GLOB SHADERS "shaders/*.vert" "shaders/*.frag")
foreach(shader ${SHADERS})
    get_filename_component(shader_name ${shader} NAME)
    add_custom_command(TARGET BlackHoleCUDA POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:BlackHoleCUDA>/shaders/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${shader}
            $<TARGET_FILE_DIR:BlackHoleCUDA>/shaders/${shader_name}
    )
endforeach()
