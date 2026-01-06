#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "track_manager.h"
#include "deformation_graph.h"


void runTrackViewer(
    const std::vector<std::string>& cam0_image_names,
    TrackManager& track_manager,
    const Eigen::Matrix3d& K,
    const std::vector<std::string>& depth_image_names,
    double depth_scale);
