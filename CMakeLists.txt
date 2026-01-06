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
set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# yaml-cpp
set(YAML_CPP_DIR "/opt/homebrew/opt/yaml-cpp")
include_directories(${YAML_CPP_DIR}/include)
link_directories(${YAML_CPP_DIR}/lib)

# pybind11 + Python 3.8
set(PYTHON_VERSION 3.8)
set(PYTHON_INCLUDE_DIR /Library/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/include/python${PYTHON_VERSION})
set(PYTHON_LIBRARY /Library/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/lib/libpython${PYTHON_VERSION}.dylib)
include_directories(${PYTHON_INCLUDE_DIR})
link_directories(/Library/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/lib)
include_directories(/Library/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/lib/python${PYTHON_VERSION}/site-packages/pybind11/include)

# Ceres Solver
find_package(Ceres REQUIRED)
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
    src/alignment_utils
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
