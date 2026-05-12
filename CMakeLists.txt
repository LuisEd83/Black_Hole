cmake_minimum_required(VERSION 3.24)
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
find_package(Threads REQUIRED)


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


set(BUILD_MODE "all" CACHE STRING "Build mode: png | gl | all")

set(SOURCES_COMMON
    cuda/comms.cu
    cuda/feedbacks.cu
    src/lodepng.cpp
    src/starmap.cpp
    src/perlin.cpp
    src/temp_and_time.cpp
    src/host.cpp
    src/engine.cpp
    main.cpp
)

set(SOURCES_PNG cuda/png_kernel.cu)
set(SOURCES_GL  cuda/gl_kernel.cu)


if(BUILD_MODE STREQUAL "png")
    set(ACTIVE_SOURCES ${SOURCES_COMMON} ${SOURCES_PNG})
elseif(BUILD_MODE STREQUAL "gl")
    set(ACTIVE_SOURCES ${SOURCES_COMMON} ${SOURCES_GL})
else()
    set(ACTIVE_SOURCES ${SOURCES_COMMON} ${SOURCES_PNG} ${SOURCES_GL})
endif()


add_executable(BlackHoleCUDA ${ACTIVE_SOURCES})

target_link_libraries(BlackHoleCUDA PRIVATE
    ${DEPS}
    CUDA::cudart
    Threads::Threads
    #pthread
)

set_target_properties(BlackHoleCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         native
    LINKER_LANGUAGE            CUDA     # deixa o NVCC fazer o link final
)
target_compile_options(BlackHoleCUDA PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G>
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
