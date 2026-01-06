#pragma once

#include <map>
#include <set>
#include <vector>
#include <Eigen/Dense>
#include "ceres_refine.h"


bool solveVINSWithBias(
    const std::map<int,Eigen::Matrix3d> &Gamma_j_map,
    const std::map<int,Eigen::Vector3d> &S_j_map,
    const std::map<std::pair<int,int>,Eigen::Vector3d> &mu_ij_map,
    const std::vector<double> &t,
    double g_mag,
    VISolution &solution);
