set(THIS pyflash)

file (GLOB SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

set (INCL
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# torch_python is needed to run python benchmarks
find_library(
   TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib"
 )
find_package(Python REQUIRED COMPONENTS Development)

# dependencies
set (DEPS
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    Python::Python
    flash
)

# Make a library
add_library(${THIS} SHARED ${SRC})

target_link_libraries(${THIS} PRIVATE ${DEPS})
target_include_directories(${THIS} PRIVATE ${INCL})

SET_TARGET_PROPERTIES (
    ${THIS} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CXX
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}"
    # CUDA_ARCHITECTURES  ${NUDFT_CUDA_ARCHITECTURES}
)
