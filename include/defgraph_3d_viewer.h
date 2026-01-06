#pragma once

#include <unordered_map>
#include <vector>
#include "deformation_graph.h"  // NodeId, NodeRest, ViscousEdge



void runDeformationGraph3D(
    const std::unordered_map<NodeId, NodeRest>& nodes_rest,
    const std::map<int, NodePositions>& nodes_per_kf,   
    const std::vector<int>& optimized_kfs,
    const std::unordered_map<int, KFPose>& cam_poses,
    const std::vector<ViscousEdge>& viscous_edges);
