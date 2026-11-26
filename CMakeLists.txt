cmake_minimum_required(VERSION 3.10)
project(defvins)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include headers
include_directories(include)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenCV
# On macOS/Homebrew keep the original hint; elsewhere rely on the default search
# path (or pass -DOpenCV_DIR=...).
if(APPLE AND NOT OpenCV_DIR AND EXISTS /opt/homebrew/opt/opencv/lib/cmake/opencv4)
    set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# yaml-cpp
set(YAML_CPP_DIR "/opt/homebrew/opt/yaml-cpp")
if(EXISTS ${YAML_CPP_DIR})
    include_directories(${YAML_CPP_DIR}/include)
    link_directories(${YAML_CPP_DIR}/lib)
else()
    find_package(yaml-cpp REQUIRED)
endif()

# pybind11 + embedded Python (used to call the C2P solver, scripts/nonmin_pose_wrapper.py)
# macOS: the original setup used the python.org 3.8 framework. It is still used
# by default when present; otherwise CMake's FindPython3 picks the system Python
# (override with -DPython3_ROOT_DIR=... or -DPython3_EXECUTABLE=...).
set(DEFVINS_LEGACY_PY_FRAMEWORK /Library/Frameworks/Python.framework/Versions/3.8)
if(APPLE AND NOT Python3_ROOT_DIR AND NOT Python3_EXECUTABLE AND EXISTS ${DEFVINS_LEGACY_PY_FRAMEWORK})
    set(Python3_ROOT_DIR ${DEFVINS_LEGACY_PY_FRAMEWORK})
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    include_directories(${pybind11_INCLUDE_DIRS})
else()
    # pybind11 installed with pip into the selected interpreter
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_PIP_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_PIP_RESULT)
    if(NOT PYBIND11_PIP_RESULT EQUAL 0)
        message(FATAL_ERROR "pybind11 not found (neither CMake package nor pip module)")
    endif()
    include_directories(${PYBIND11_PIP_INCLUDE})
endif()

# Ceres Solver (>= 2.1 required: the code uses ceres::Manifold)
find_package(Ceres 2.1 REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)

# Executable and source files
add_executable(defvins
    src/main.cpp
    src/ceres_refine.cpp
    src/bg_small_angle.cpp
    src/bg_optimization.cpp
    src/mandala_io.cpp
    src/c2p_wrapper.cpp
    src/camera_utils.cpp
    src/feature_tracking.cpp
    src/frame_processor.cpp
    src/feature_tracker.cpp
    src/track_manager.cpp
    src/visualize_tracking.cpp
    src/utils.cpp
    src/viewer.cpp
    src/defgraph_3d_viewer.cpp
    src/vins_closed_form.cpp
    src/deformation_graph.cpp
    src/alignment_utils.cpp
)

target_include_directories(defvins PRIVATE
    ${Pangolin_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(defvins
    yaml-cpp
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${PYTHON_LIBRARY}
    ${CERES_LIBRARIES}
)

# modo debug
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined -fno-omit-frame-pointer")

# mkdir -p build && cd build
# cmake -DCMAKE_BUILD_TYPE=Release ..
# cmake --build . -j

# cmake .. \
#  -DCMAKE_BUILD_TYPE=Release \
#  -DPangolin_DIR=/Users/samucerezo/dev/Pangolin/build

# make -j8
