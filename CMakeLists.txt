cmake_minimum_required(VERSION 3.12) # Standard version
project(spex)

set(CMAKE_CXX_STANDARD 17)

# 1. Find the pybind11 package
find_package(pybind11 REQUIRED)

# 2. Define the Python Module
# This creates 'stl.so' (Linux/Mac) or 'stl.pyd' (Windows) that you can 'import stl' in Python
pybind11_add_module(stl_binding spex.cpp)
target_link_libraries(stl_binding PRIVATE pybind11::headers)
target_include_directories(stl_binding PRIVATE include)
