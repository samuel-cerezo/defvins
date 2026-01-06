// lie_utils.h
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

namespace lie {

    /// Returns the skew-symmetric matrix associated with a 3D vector.
    /// This is used in cross product and SO(3) exponential/logarithm maps.
    ///
    /// Input:
    /// - v: 3x1 vector
    ///
    /// Output:
    /// - 3x3 skew-symmetric matrix
    inline Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m <<     0, -v(2),  v(1),
            v(2),     0, -v(0),
            -v(1),  v(0),     0;
        return m;
    }

    /// Exponential map from so(3) to SO(3).
    /// Converts a rotation vector (axis-angle) into a 3x3 rotation matrix.
    ///
    /// Input:
    /// - w: 3x1 rotation vector
    ///
    /// Output:
    /// - 3x3 rotation matrix in SO(3)
    inline Eigen::Matrix3d ExpMap(const Eigen::Vector3d& w) {
        double theta = w.norm();
        if (theta < 1e-8) {
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Vector3d axis = w / theta;
        Eigen::Matrix3d wx = skew_symmetric(axis);

        return Eigen::Matrix3d::Identity()
            + wx * std::sin(theta)
            + wx * wx * (1.0 - std::cos(theta));
    }

    /// Logarithmic map from SO(3) to so(3) - original version.
    /// Converts a 3x3 rotation matrix into a rotation vector (axis-angle).
    inline Eigen::Vector3d LogMap(const Eigen::Matrix3d& R) {
        if ((R - Eigen::Matrix3d::Identity()).norm() < 1e-10) {
            return Eigen::Vector3d::Zero();
        }

        double cos_theta = (R.trace() - 1.0) / 2.0;
        cos_theta = std::clamp(cos_theta, -1.0, 1.0);
        double theta = std::acos(cos_theta);

        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        if (std::abs(sin_theta) < 1e-8) {
            sin_theta = 1e-8;
        }

        Eigen::Matrix3d lnR = (theta / (2.0 * sin_theta)) * (R - R.transpose());
        return Eigen::Vector3d(lnR(2, 1), lnR(0, 2), lnR(1, 0));
    }

    /// Robust LogMap using Eigen::AngleAxis (new version)
    inline Eigen::Vector3d LogMapRobust(const Eigen::Matrix3d& R) {
        Eigen::AngleAxisd aa(R);
        return aa.angle() * aa.axis();
    }

    /// Templated skew-symmetric for autodiff types (e.g., ceres::Jet)
    template <typename T>
    inline Eigen::Matrix<T, 3, 3> skewTemplated(const Eigen::Matrix<T, 3, 1>& v) {
        Eigen::Matrix<T, 3, 3> m;
        m <<     T(0), -v(2),  v(1),
              v(2),     T(0), -v(0),
             -v(1),  v(0),     T(0);
        return m;
    }

    /// Templated exponential map from so(3) to SO(3)
    template <typename Derived>
    inline Eigen::Matrix<typename Derived::Scalar, 3, 3> ExpMapTemplated(const Eigen::MatrixBase<Derived>& w) {
        using T = typename Derived::Scalar;
        T theta = w.norm();
        if (theta < T(1e-8)) {
            return Eigen::Matrix<T, 3, 3>::Identity();
        }

        Eigen::Matrix<T, 3, 1> axis = w / theta;

        Eigen::Matrix<T, 3, 3> wx;
        wx << T(0), -axis(2),  axis(1),
            axis(2), T(0), -axis(0),
            -axis(1), axis(0), T(0);

        return Eigen::Matrix<T, 3, 3>::Identity()
            + wx * sin(theta)
            + wx * wx * (T(1.0) - cos(theta));
    }

    /// Templated logarithmic map from SO(3) to so(3) - original version
    template <typename T>
    inline Eigen::Matrix<T, 3, 1> LogMapTemplated(const Eigen::Matrix<T, 3, 3>& R) {
        Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();
        T norm_R = (R - I).norm();
        if (norm_R < T(1e-10)) {
            return Eigen::Matrix<T, 3, 1>::Zero();
        }

        T cos_theta = (R.trace() - T(1.0)) / T(2.0);
        cos_theta = std::max(std::min(cos_theta, T(1.0)), T(-1.0));
        T theta = acos(cos_theta);

        T sin_theta = sqrt(T(1.0) - cos_theta * cos_theta);
        if (abs(sin_theta) < T(1e-8)) sin_theta = T(1e-8);

        Eigen::Matrix<T, 3, 3> lnR = (theta / (T(2.0) * sin_theta)) * (R - R.transpose());
        return Eigen::Matrix<T, 3, 1>(lnR(2, 1), lnR(0, 2), lnR(1, 0));
    }

    /// Templated robust LogMap using quaternion-based method
    template <typename T>
    inline Eigen::Matrix<T, 3, 1> LogMapTemplatedRobust(const Eigen::Matrix<T, 3, 3>& R) {
        Eigen::Quaternion<T> q(R);
        T norm_v = sqrt(q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
        T angle = T(2.0) * atan2(norm_v, q.w());
        if (norm_v < T(1e-10)) {
            return Eigen::Matrix<T, 3, 1>::Zero();
        } else {
            return angle * Eigen::Matrix<T, 3, 1>(q.x(), q.y(), q.z()) / norm_v;
        }
    }

} // namespace lie
