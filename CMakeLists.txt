cmake_minimum_required(VERSION 3.14)

# ───────────────────────────────────────────────────────────────
# Slang compiler
# ───────────────────────────────────────────────────────────────
find_program(SLANGC_EXECUTABLE slangc)
if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found. Install the Slang compiler from https://github.com/shader-slang/slang/releases")
endif()

# ───────────────────────────────────────────────────────────────
# Shader compilation helper
# ───────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────
# Project
# ───────────────────────────────────────────────────────────────
project(BlackHoleSim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ───────────────────────────────────────────────────────────────
# Dependencies
# ───────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────
# Source files
# ───────────────────────────────────────────────────────────────
set(SOURCES
    main.cpp
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

# ───────────────────────────────────────────────────────────────
# Executable
# ───────────────────────────────────────────────────────────────
add_executable(black_hole_sim ${SOURCES})

add_slang_shader_target(slang_shaders SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(black_hole_sim slang_shaders)

# ───────────────────────────────────────────────────────────────
# Compile definitions
# ───────────────────────────────────────────────────────────────
target_compile_definitions(black_hole_sim PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)

# ───────────────────────────────────────────────────────────────
# Include directories
# ───────────────────────────────────────────────────────────────
target_include_directories(black_hole_sim PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${Vulkan_INCLUDE_DIR}
)

if(STB_INCLUDE_DIR)
    target_include_directories(black_hole_sim PRIVATE ${STB_INCLUDE_DIR})
endif()

# ───────────────────────────────────────────────────────────────
# Link libraries
# ───────────────────────────────────────────────────────────────
target_link_libraries(black_hole_sim PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
)

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

# ───────────────────────────────────────────────────────────────
# Output directory
# ───────────────────────────────────────────────────────────────
set_target_properties(black_hole_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build
)

# ───────────────────────────────────────────────────────────────
# Targets de execução
# ───────────────────────────────────────────────────────────────

# Roda a simulação interativa normalmente
add_custom_target(run
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/build/black_hole_sim
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS black_hole_sim
    COMMENT "Rodando simulação..."
)

# Exporta dados do disco e plota automaticamente
add_custom_target(disk
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/build/black_hole_sim --disk 200
    COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/python/plot_disk.py --save
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS black_hole_sim
    COMMENT "Exportando dados do disco e gerando plots..."
)

# Exporta frame PNG
add_custom_target(export
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/build/black_hole_sim --export icons/output.png 1920 1080
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS black_hole_sim
    COMMENT "Exportando frame PNG..."
)