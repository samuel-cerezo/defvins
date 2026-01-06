#pragma once
#include <vector>
#include <Eigen/Dense>

struct ImuSample {
  double t;
  Eigen::Vector3d omega;  // rad/s
  Eigen::Vector3d acc;    // m/s^2
};

struct PreintMeas {
  Eigen::Matrix3d dR;     // ΔR_ij
  Eigen::Vector3d dv;     // Δv_ij
  Eigen::Vector3d dp;     // Δp_ij
  double dt;              // Δt_ij
  // (opcional) covariancias
};

/// Preintegra entre dos tiempos usando bg, ba y g actuales (simplificado)
PreintMeas Preintegrate(const std::vector<ImuSample>& imu, double t_i, double t_j,
                        const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
                        const Eigen::Vector3d& g);
