#include "bg_small_angle.h"
#include "lie_utils.h"
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <iostream>

struct IMURotationResidual {
    IMURotationResidual(const Eigen::Matrix3d& Rij,
                        const Eigen::MatrixXd& omega_all,
                        const Eigen::Matrix3d& R_cam_imu,
                        double deltat,
                        const Eigen::Matrix3d& R_cam,
                        const Eigen::Matrix3d& R_imu)
        : Rij_(Rij), omega_all_(omega_all), R_cam_imu_(R_cam_imu), deltat_(deltat), R_cam_(R_cam), R_imu_(R_imu) {}

    template <typename T>
    bool operator()(const T* const bg, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        Mat3T Rpreint = Mat3T::Identity();

        for (int i = 0; i < omega_all_.cols(); ++i) {
            Vec3T omega_i;
            for (int j = 0; j < 3; ++j) {
                omega_i(j) = T(omega_all_(j, i));
            }
            Vec3T omega = omega_i - Eigen::Map<const Vec3T>(bg);
            Rpreint = Rpreint * lie::ExpMapTemplated(omega * T(deltat_));
        }
        // retrieving transf matrices
        Mat3T R_cam_imu_T = R_cam_imu_.cast<T>();
        Mat3T R_cam_T = R_cam_.cast<T>();
        Mat3T R_imu_T = R_imu_.cast<T>();
        Mat3T Rij_T = Rij_.cast<T>();

        // Compose integrated and measured rotation in the same reference frame
        Mat3T R_integrated_T = R_imu_T * Rpreint * R_imu_T.transpose();
        Mat3T R_measured_T = R_cam_T * Rij_T * R_cam_T.transpose();
        Mat3T R_error = R_integrated_T.transpose() * R_cam_T * Rij_T * R_cam_T.transpose();
        
        Vec3T res = lie::LogMapTemplated(R_error);

        residuals[0] = res[0];
        residuals[1] = res[1];
        residuals[2] = res[2];
        return true;
    }

    const Eigen::Matrix3d Rij_;
    const Eigen::MatrixXd omega_all_;
    const Eigen::Matrix3d R_cam_imu_;
    const Eigen::Matrix3d R_cam_;
    const Eigen::Matrix3d R_imu_;
    const double deltat_;
};

std::tuple<Eigen::Vector3d, double, ceres::Solver::Summary>
bg_optimization (const std::vector<Eigen::Vector3d>& omega_all_vec,
                      double deltat,
                      const Eigen::Matrix3d& Rij,
                      const Eigen::Vector3d& bg0,
                      const Eigen::Matrix4d& tbodycam,
                      const Eigen::Matrix4d& tbodyimu)
{
    Eigen::Matrix3d R_cam = tbodycam.block<3, 3>(0, 0);
    Eigen::Matrix3d R_imu = tbodyimu.block<3, 3>(0, 0);
    Eigen::Matrix3d R_cam_imu = R_cam * R_imu.transpose();

    ceres::Problem problem;
    Eigen::Vector3d bg = bg0;

    Eigen::MatrixXd omega_all(3, omega_all_vec.size());
    for (size_t i = 0; i < omega_all_vec.size(); ++i) {
        omega_all.col(i) = omega_all_vec[i];
    }

    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<IMURotationResidual, 3, 3>(
            new IMURotationResidual(Rij, omega_all, R_cam_imu, deltat, R_cam, R_imu));

    problem.AddResidualBlock(cost_function, nullptr, bg.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 10000;
    options.function_tolerance = 1e-16;
    options.gradient_tolerance = 1e-16;
    options.parameter_tolerance = 1e-16;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    double final_cost = problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, nullptr);
    return {bg, final_cost, summary};
}
