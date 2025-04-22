set(THIS flash)

file(GLOB_RECURSE SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

set (DEPS ${TORCH_LIBRARIES})

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
# find_package(Thrust 2 REQUIRED CONFIG)
# thrust_create_target(Thrust)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is not defined.\n"
        "Please set it correctly because I don't want to guess.\n"
        "You can consult: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/. ")
endif()

add_library(${THIS} ${SRC})
target_link_libraries(${THIS} PRIVATE ${DEPS})

SET_TARGET_PROPERTIES (
    ${THIS} PROPERTIES
    CXX_STANDARD 20
    CUDA_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CXX
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}"
    # CUDA_ARCHITECTURES  ${NUDFT_CUDA_ARCHITECTURES}
)
