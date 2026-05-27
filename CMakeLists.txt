cmake_minimum_required(VERSION 3.24)

# ── Plataformas suportadas ────────────────────────────────────────────────────
# Linux   : GCC/Clang + CUDA Toolkit + GLFW/GLEW via apt/pacman
# macOS   : Clang (CPU backend, sem CUDA) + GLFW/GLEW via Homebrew
#           (Apple Silicon e Intel-Mac: macOS 10.14+ removeu suporte CUDA)
# Windows : MSVC (Visual Studio 2019/2022) + CUDA Toolkit + GLFW/GLEW via vcpkg
#           Use: cmake -B build -G "Visual Studio 17 2022" -A x64
# ─────────────────────────────────────────────────────────────────────────────

if(APPLE)
    project(BlackHoleSim LANGUAGES CXX)
else()
    project(BlackHoleSim LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT APPLE)
    set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GLEW        REQUIRED)
find_package(glfw3       REQUIRED)
find_package(glm         REQUIRED)
find_package(OpenGL      REQUIRED)
if(NOT APPLE)
    find_package(CUDAToolkit REQUIRED)
endif()
find_package(Threads     REQUIRED)

set(DEPS glfw GLEW::GLEW glm::glm OpenGL::GL)

# ── Fontes comuns a todas as plataformas ─────────────────────────────────────


set(COMMON_SOURCES
    src/lodepng.cpp
    src/starmap.cpp
    src/perlin.cpp
    src/temp_and_time.cpp
    src/host.cpp
    cuda/feedbacks.cu
    cuda/comms.cu

)

set(SOURCES_PNG 
    cuda/png_kernel.cu 
    png.cpp
)

set(SOURCES_GL  
    src/engine.cpp
    cuda/gl_kernel.cu 
    gl.cpp
)


# ── Fontes específicas por plataforma ────────────────────────────────────────
#
# macOS  : cpu_raytrace.cpp substitui os kernels CUDA
# outros : .cu compilados pelo NVCC; linker final é o NVCC


if(APPLE)
    set(PLATFORM_SOURCES src/cpu_raytrace.cpp)

else()
        
    if(BUILD_MODE STREQUAL "png")
        set(SIM_TYPE ${SOURCES_PNG})

    elseif(BUILD_MODE STREQUAL "gl")
        set(SIM_TYPE ${SOURCES_GL})

    else()
        set(SIM_TYPE ${SOURCES_PNG} ${SOURCES_GL})

    endif()

endif()

add_executable(BlackHoleCUDA
    ${COMMON_SOURCES}
    ${PLATFORM_SOURCES}
    ${SIM_TYPE}
)


# ── Link e propriedades por plataforma ───────────────────────────────────────
if(APPLE)
    target_link_libraries(BlackHoleCUDA PRIVATE
        ${DEPS}
        Threads::Threads
    )
else()
    target_link_libraries(BlackHoleCUDA PRIVATE
        ${DEPS}
        CUDA::cudart
        Threads::Threads
    )
    set_target_properties(BlackHoleCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES         native
        LINKER_LANGUAGE            CUDA
    )
    get_target_property(CUDA_INCLUDE_DIRS CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(BlackHoleCUDA SYSTEM PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CUDA_INCLUDE_DIRS}>
    )

    target_compile_options(BlackHoleCUDA PRIVATE
         $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

endif()

# ── M_PI: garantir que está definido no MSVC ─────────────────────────────────
if(MSVC)
    target_compile_definitions(BlackHoleCUDA PRIVATE _USE_MATH_DEFINES)
endif()

# ── Copia shaders para o diretório do executável ─────────────────────────────
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

# ── Copia data/ para o diretório do executável ───────────────────────────────
file(GLOB DATA_FILES "${CMAKE_SOURCE_DIR}/data/*")
foreach(data_file ${DATA_FILES})
    get_filename_component(data_name ${data_file} NAME)
    add_custom_command(TARGET BlackHoleCUDA POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:BlackHoleCUDA>/data/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${data_file}
            $<TARGET_FILE_DIR:BlackHoleCUDA>/data/${data_name}
    )
endforeach()
