cmake_minimum_required(VERSION 3.24)

# ───────────────────────────────────────────────────────────────
# Build mode selection
# ───────────────────────────────────────────────────────────────
set(BUILD_MODE "video" CACHE STRING "Build mode: png (static CUDA) or video (Vulkan)")
set_property(CACHE BUILD_MODE PROPERTY STRINGS png video)

# ───────────────────────────────────────────────────────────────
# Project
# ───────────────────────────────────────────────────────────────
if(BUILD_MODE STREQUAL "png")
    project(BlackHoleSim LANGUAGES CXX CUDA)
else()
    project(BlackHoleSim LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(BUILD_MODE STREQUAL "png")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ───────────────────────────────────────────────────────────────
# PNG mode — static CUDA simulation
# ───────────────────────────────────────────────────────────────
if(BUILD_MODE STREQUAL "png")

    find_package(GLEW        REQUIRED)
    find_package(glfw3       REQUIRED)
    find_package(glm         REQUIRED)
    find_package(OpenGL      REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    find_package(Threads     REQUIRED)

    set(SOURCES
        cuda/feedbacks.cu
        cuda/comms.cu
        cuda/geod.cu
        cuda/effects.cu
        cuda/raytrace.cu
        cuda/png_kernel.cu
        src/lodepng.cpp
        src/starmap.cpp
        src/perlin.cpp
        src/temp_and_time.cpp
        src/host.cpp
        png.cpp
    )

    add_executable(black_hole_sim ${SOURCES})

    # lodepng is third-party: silence its warnings
    set_source_files_properties(src/lodepng.cpp PROPERTIES COMPILE_FLAGS "-w")

    target_link_libraries(black_hole_sim PRIVATE
        glfw
        GLEW::GLEW
        glm::glm
        OpenGL::GL
        CUDA::cudart
        Threads::Threads
    )

    set_target_properties(black_hole_sim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES         native
        LINKER_LANGUAGE            CUDA
        RUNTIME_OUTPUT_DIRECTORY   ${CMAKE_CURRENT_LIST_DIR}/build
    )

    get_target_property(CUDA_INCLUDE_DIRS CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(black_hole_sim SYSTEM PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CUDA_INCLUDE_DIRS}>
    )

    target_include_directories(black_hole_sim PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/headers
    )

    target_compile_options(black_hole_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    # ── M_PI: garantir que está definido no MSVC ─────────────────
    if(MSVC)
        target_compile_definitions(black_hole_sim PRIVATE _USE_MATH_DEFINES)
    endif()

# ───────────────────────────────────────────────────────────────
# VIDEO mode — Vulkan dynamic simulation
# ───────────────────────────────────────────────────────────────
elseif(BUILD_MODE STREQUAL "video")

    # ───────────────────────────────────────────────────────────
    # Slang compiler
    # ───────────────────────────────────────────────────────────
    find_program(SLANGC_EXECUTABLE slangc)
    if(NOT SLANGC_EXECUTABLE)
        message(FATAL_ERROR "slangc not found. Install the Slang compiler from https://github.com/shader-slang/slang/releases")
    endif()

    # ───────────────────────────────────────────────────────────
    # Shader compilation helper
    # ───────────────────────────────────────────────────────────
    function(add_slang_shader_target TARGET)
        cmake_parse_arguments(SHADER "" "" "SOURCES" ${ARGN})
        set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
        set(ENTRY_POINTS -entry vertMain -entry fragMain)

        add_custom_command(
            OUTPUT ${SHADERS_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
        )

        add_custom_command(
            OUTPUT ${SHADERS_DIR}/graphics.spv
            COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES}
                    -target spirv
                    -profile spirv_1_4
                    -emit-spirv-directly
                    -fvk-use-entrypoint-name
                    ${ENTRY_POINTS}
                    -o ${SHADERS_DIR}/graphics.spv
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            DEPENDS ${SHADER_SOURCES}
            COMMENT "Compiling Graphics Shaders"
            VERBATIM
        )

        set(COMPUTE_SLANG_SOURCE ${CMAKE_CURRENT_LIST_DIR}/shaders/compute.slang)
        set(COMPUTE_SLANG_DEPS
            ${CMAKE_CURRENT_LIST_DIR}/shaders/compute.slang
            ${CMAKE_CURRENT_LIST_DIR}/shaders/constants.slang
            ${CMAKE_CURRENT_LIST_DIR}/shaders/geodesic.slang
            ${CMAKE_CURRENT_LIST_DIR}/shaders/effects.slang
        )

        add_custom_command(
            OUTPUT ${SHADERS_DIR}/compute.spv
            COMMAND ${SLANGC_EXECUTABLE} ${COMPUTE_SLANG_SOURCE}
                    -target spirv
                    -profile spirv_1_4
                    -emit-spirv-directly
                    -fvk-use-entrypoint-name
                    -entry computeMain
                    -o ${SHADERS_DIR}/compute.spv
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            DEPENDS ${COMPUTE_SLANG_DEPS}
            COMMENT "Compiling Compute Shaders"
            VERBATIM
        )

        add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/graphics.spv ${SHADERS_DIR}/compute.spv)
    endfunction()

    # ───────────────────────────────────────────────────────────
    # Dependencies
    # ───────────────────────────────────────────────────────────
    set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
    set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
    set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")

    find_package(Vulkan REQUIRED)
    find_package(glfw3 3.3 REQUIRED)
    find_package(glm REQUIRED)

    # STB is header-only; try to find it in common system paths
    find_path(STB_INCLUDE_DIR
        NAMES stb_image.h stb_image_write.h
        PATHS
            /usr/include/stb
            /usr/local/include/stb
            ${CMAKE_CURRENT_LIST_DIR}/third_party/stb
            ${CMAKE_CURRENT_LIST_DIR}/external/stb
        DOC "STB image headers directory"
    )
    if(NOT STB_INCLUDE_DIR)
        message(WARNING "STB headers not found in standard paths. Please install stb or place it in a 'third_party/stb' directory. The build may fail if stb_image.h is not on the include path.")
    endif()

    # ───────────────────────────────────────────────────────────
    # Source files
    # ───────────────────────────────────────────────────────────
    set(SOURCES
        video.cpp
        src/test_app.cpp
        src/vk_init/compute_pipeline.cpp
        src/vk_init/graphics_pipeline.cpp
        src/vk_init/presentation.cpp
        src/vk_init/setup.cpp
        src/vk_main/render.cpp
        src/stb_image_write_impl.cpp
    )

    set(SHADER_SLANG_SOURCES
        shaders/graphics.slang
    )

    # ───────────────────────────────────────────────────────────
    # Executable
    # ───────────────────────────────────────────────────────────
    add_executable(black_hole_sim ${SOURCES})

    add_slang_shader_target(slang_shaders SOURCES ${SHADER_SLANG_SOURCES})
    add_dependencies(black_hole_sim slang_shaders)

    # ───────────────────────────────────────────────────────────
    # Compile definitions
    # ───────────────────────────────────────────────────────────
    target_compile_definitions(black_hole_sim PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )

    # ───────────────────────────────────────────────────────────
    # Include directories
    # ───────────────────────────────────────────────────────────
    target_include_directories(black_hole_sim PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${Vulkan_INCLUDE_DIR}
    )

    if(STB_INCLUDE_DIR)
        target_include_directories(black_hole_sim PRIVATE ${STB_INCLUDE_DIR})
    endif()

    # ───────────────────────────────────────────────────────────
    # Link libraries
    # ───────────────────────────────────────────────────────────
    target_link_libraries(black_hole_sim PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
    )

    # ───────────────────────────────────────────────────────────
    # Output directory
    # ───────────────────────────────────────────────────────────
    set_target_properties(black_hole_sim PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build
    )

# ───────────────────────────────────────────────────────────────
# Unknown mode
# ───────────────────────────────────────────────────────────────
else()
    message(FATAL_ERROR "Unknown BUILD_MODE '${BUILD_MODE}'. Use 'png' or 'video'.")
endif()

# ───────────────────────────────────────────────────────────────
# Compiler warnings (strict)
# ───────────────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(black_hole_sim PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-pedantic>
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(black_hole_sim PRIVATE
            -Wmisleading-indentation
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
        )
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(black_hole_sim PRIVATE
        /W4
        /WX
        /permissive-
        /w14242
        /w14254
        /w14263
        /w14265
        /w14287
        /we4289
        /w14296
        /w14311
        /w14545
        /w14546
        /w14547
        /w14549
        /w14555
        /w14619
        /w14640
        /w14826
        /w14905
        /w14906
        /w14928
    )
endif()

# ───────────────────────────────────────────────────────────────
# Platform-specific
# ───────────────────────────────────────────────────────────────
if(APPLE)
    # macOS with MoltenVK may need extra framework linking
    target_link_libraries(black_hole_sim PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreFoundation"
    )
endif()
