#include "alignment_utils.h"


Eigen::Matrix4d poseToHom(const Eigen::Quaterniond &q,
                          const Eigen::Vector3d &t)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3,3>() = q.normalized().toRotationMatrix();
    T.topRightCorner<3,1>() = t;
    return T;
}



void homToPose(const Eigen::Matrix4d& T, 
                Eigen::Quaterniond &q,
                Eigen::Vector3d &t)
{

    t = T.topRightCorner<3,1>();
    Eigen::Matrix3d R = T.topLeftCorner<3,3>();

    q = Eigen::Quaterniond(R);
    q.normalize();
}


std::vector<Eigen::Quaterniond> transformQuaternionsRight (const std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses,
                                                   const Eigen::Matrix4d& transform)
{
    std::vector<Eigen::Quaterniond> transformed_quaternions;
    transformed_quaternions.reserve(poses.size());

    for(const auto& pose : poses){
        const Eigen::Quaterniond& q_in = pose.first;
        const Eigen::Vector3d& t_in = pose.second;

        Eigen::Matrix4d T = poseToHom(q_in, t_in);
        Eigen::Matrix4d T_transformed = T * transform;      // right product
        Eigen::Quaterniond q_out;
        Eigen::Vector3d t_out;
        
        homToPose(T_transformed, q_out, t_out);
        transformed_quaternions.emplace_back(q_out);
    }

    return transformed_quaternions;
}



std::vector<Eigen::Quaterniond> transformQuaternionsLeft (const std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses,
                                                   const Eigen::Matrix4d& transform)
{
    std::vector<Eigen::Quaterniond> transformed_quaternions;
    transformed_quaternions.reserve(poses.size());

    for(const auto& pose : poses){
        const Eigen::Quaterniond& q_in = pose.first;
        const Eigen::Vector3d& t_in = pose.second;

        Eigen::Matrix4d T = poseToHom(q_in, t_in);
        Eigen::Matrix4d T_transformed = transform * T;      // left product
        Eigen::Quaterniond q_out;
        Eigen::Vector3d t_out;
        
        homToPose(T_transformed, q_out, t_out);
        transformed_quaternions.emplace_back(q_out);
    }

    return transformed_quaternions;
}


std::vector<Eigen::Vector3d> transformPositionsLeft (const std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses,
                                                   const Eigen::Matrix4d& transform)
{
    std::vector<Eigen::Vector3d> transformed_positions;
    transformed_positions.reserve(poses.size());

    for(const auto& pose : poses){
        const Eigen::Quaterniond& q = pose.first;
        const Eigen::Vector3d& t = pose.second;

        Eigen::Matrix4d T = poseToHom(q,t);
        Eigen::Matrix4d T_transformed = transform * T;      // left product
        transformed_positions.emplace_back(T_transformed.topRightCorner<3,1>());
    }

    return transformed_positions;
}

std::vector<Eigen::Vector3d> transformPositionsRight (const std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses,
                                                   const Eigen::Matrix4d& transform)
{
    std::vector<Eigen::Vector3d> transformed_positions;
    transformed_positions.reserve(poses.size());

    for(const auto& pose : poses){
        const Eigen::Quaterniond& q = pose.first;
        const Eigen::Vector3d& t = pose.second;

        Eigen::Matrix4d T = poseToHom(q,t);
        Eigen::Matrix4d T_transformed = T * transform;      // right product
        transformed_positions.emplace_back(T_transformed.topRightCorner<3,1>());
    }

    return transformed_positions;
}

