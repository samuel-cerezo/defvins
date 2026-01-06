#ifndef C2P_WRAPPER_H
#define C2P_WRAPPER_H

#include <Eigen/Dense>
#include <vector>

struct C2PResult {
    Eigen::Matrix3d E;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    bool is_optimal;
    bool is_pure_rot;
};

C2PResult run_c2p(const std::vector<Eigen::Vector3d>& f0_inliers,
                  const std::vector<Eigen::Vector3d>& f1_inliers,
                  int frame_idx);

#endif // C2P_WRAPPER_H
