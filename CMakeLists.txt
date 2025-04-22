cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FlashAttention)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(flash)

set(DEPS "${TORCH_LIBRARIES} flash")

add_executable(runner main.cpp)
target_link_libraries(runner "${TORCH_LIBRARIES}" flash)
set_property(TARGET runner PROPERTY CXX_STANDARD 20)

add_subdirectory(pyflash)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${PROJECT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()
