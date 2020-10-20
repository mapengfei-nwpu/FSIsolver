cmake_minimum_required(VERSION 3.5)
project(test_function)

# Find pybind11
find_package(pybind11 REQUIRED)
# Find dolfin
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})
# Compiler configure
# add_definitions("-Wall -g")
add_definitions("-O3")
# Target libraries
# Add pybin11 module
pybind11_add_module(IB SHARED ParticleInterpolation.cpp)

# MESSAGE(STATUS "This is SOURCE dir " ${test_function_SOURCE_DIR})
target_link_libraries(IB PRIVATE dolfin  ${test_function_SOURCE_DIR}/libparticles.so)