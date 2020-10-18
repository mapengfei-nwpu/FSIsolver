cmake_minimum_required(VERSION 3.5)
project(test_function)

# Find dolfin
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})
# Executable
add_executable(${PROJECT_NAME} particles.cpp)
# Compiler configure
add_definitions("-O3")
# Target libraries
MESSAGE(STATUS "This is SOURCE dir " ${test_function_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE dolfin  ${test_function_SOURCE_DIR}/libparticles.so)