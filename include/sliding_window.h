#pragma once
#include <deque>
#include <Eigen/Dense>

struct KFState {
  int    k;                    // keyframe index
  double t;                    // timestamp (cam)
  Eigen::Matrix3d R;           // world->cam rotation (o cam->world, pero sé constante)
  Eigen::Vector3d p;           // position
  Eigen::Vector3d v;           // velocity
};

struct GlobalParams {
  Eigen::Vector3d g;           // gravity in world
  Eigen::Vector3d bg;          // gyro bias (constant in window, for now)
  Eigen::Vector3d ba;          // accel bias (constant in window, optimized)
};

struct Window10 {
  std::deque<KFState> kfs;     // up to 10
  GlobalParams params;
  // prior gaussiano para la ventanita (marginalización)
  Eigen::MatrixXd A_prior;     // puede estar vacío al principio
  Eigen::VectorXd b_prior;
};

/// Administra push/pop de keyframes y prepara slices IMU entre consecutivos
class WindowManager {
public:
  explicit WindowManager(size_t max_kf = 10) : max_kf_(max_kf) {}
  void pushKeyframe(const KFState& kf);
  bool full() const { return kfs().size() >= max_kf_; }
  const std::deque<KFState>& kfs() const { return win_.kfs; }
  std::deque<KFState>& kfs() { return win_.kfs; }
  Window10& window() { return win_; }

  // crea/actualiza prior tras marginalizar el más viejo
  void setPrior(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) { win_.A_prior=A; win_.b_prior=b; }

private:
  size_t max_kf_;
  Window10 win_;
};
