#include "vins_closed_form.h"
#include <Eigen/SVD>
#include <iostream>

bool solveVINSWithBias(
    const std::map<int,Eigen::Matrix3d> &Gamma_j_map,
    const std::map<int,Eigen::Vector3d> &S_j_map,
    const std::map<std::pair<int,int>,Eigen::Vector3d> &mu_ij_map,
    const std::vector<double> &t,
    double g_mag,
    VISolution &solution)
{
    std::set<int> feature_set, image_set;
    for (const auto &kv : mu_ij_map) {
        feature_set.insert(kv.first.first);
        image_set.insert(kv.first.second);
    }
    int N = static_cast<int>(feature_set.size());
    int m = static_cast<int>(image_set.size());
    if (m < 2 || N == 0 || t.size() != static_cast<size_t>(m) || t[0] != 0.0)
        return false;

    std::map<int,int> feat_to_idx, img_to_idx;
    int idx=0;
    for(int fi : feature_set) feat_to_idx[fi] = idx++;
    idx=0;
    for(int ji : image_set)  img_to_idx[ji] = idx++;

    int dimX = 9 + N*m;
    int numEq = 3 * (m-1) * N;
    Eigen::MatrixXd Xi = Eigen::MatrixXd::Zero(numEq, dimX);
    Eigen::VectorXd b  = Eigen::VectorXd::Zero(numEq);

    int row = 0;
    for(int j_orig = 1; j_orig < m; ++j_orig) {
        int j   = j_orig;
        double tj  = t[j];
        double tj2 = tj * tj;
        Eigen::Vector3d S_j     = S_j_map.at(j_orig);
        Eigen::Matrix3d Gamma_j = Gamma_j_map.at(j_orig);

        // ecuación para la característica 0
        {
            Xi.block<3,3>(row, 0) = -0.5 * tj2 * Eigen::Matrix3d::Identity();
            Xi.block<3,3>(row, 3) = -tj  * Eigen::Matrix3d::Identity();
            Xi.block<3,3>(row, 6) =  Gamma_j;

            auto f0 = *feature_set.begin();
            auto i0 = *image_set.begin();

            auto mu00 = mu_ij_map.at({f0, i0});
            auto mu0j = mu_ij_map.at({f0, j_orig});
            int idxLambda00 = 9 + feat_to_idx[f0] * m + 0;
            int idxLambda0j = 9 + feat_to_idx[f0] * m + j;

            Xi.block<3,1>(row, idxLambda00) =  mu00;
            Xi.block<3,1>(row, idxLambda0j) = -mu0j;
            b.segment<3>(row) = S_j;
            row += 3;
        }

        // ecuaciones para i=1..N-1
        auto f0 = *feature_set.begin();
        auto i0 = *image_set.begin();
        for(auto fi_it = std::next(feature_set.begin()); fi_it != feature_set.end(); ++fi_it) {
            int i = feat_to_idx[*fi_it];

            int idx00 = 9 + feat_to_idx[f0] * m + 0;
            int idx0j = 9 + feat_to_idx[f0] * m + j;
            int idxi0 = 9 + i * m + 0;
            int idxij = 9 + i * m + j;

            Xi.block<3,1>(row, idx00) =  mu_ij_map.at({f0,i0});
            Xi.block<3,1>(row, idx0j) = -mu_ij_map.at({f0,j_orig});
            Xi.block<3,1>(row, idxi0) = -mu_ij_map.at({*fi_it,i0});
            Xi.block<3,1>(row, idxij) =  mu_ij_map.at({*fi_it,j_orig});
            b.segment<3>(row) = Eigen::Vector3d::Zero();
            row += 3;
        }
    }

    Eigen::VectorXd X;
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            Xi, Eigen::ComputeThinU | Eigen::ComputeThinV);
        X = svd.solve(b);
    }

    Eigen::MatrixXd Hess = Xi.transpose() * Xi;
    Eigen::VectorXd gss  = Xi.transpose() * b;

    const int nx  = 9;
    const int Kss = Xi.cols() - nx;
    Eigen::MatrixXd Ass  = Hess.topLeftCorner(nx, nx);
    Eigen::MatrixXd Jss  = Hess.topRightCorner(nx, Kss);
    Eigen::MatrixXd Lss  = Hess.bottomRightCorner(Kss, Kss);
    Eigen::VectorXd bxss = gss.head(nx);
    Eigen::VectorXd blss = gss.tail(Kss);

    Eigen::MatrixXd L_reg = Lss;
    L_reg.diagonal().array() += 1e-9;
    Eigen::LDLT<Eigen::MatrixXd> ldltL(L_reg);
    Eigen::MatrixXd V_Li_Jt = ldltL.solve(Jss.transpose());
    Eigen::VectorXd y_Li_bl = ldltL.solve(blss);
    Eigen::MatrixXd Hs      = Ass - Jss * V_Li_Jt;
    Eigen::VectorXd gs      = bxss - Jss * y_Li_bl;

    Eigen::VectorXd x(nx);
    Eigen::LDLT<Eigen::MatrixXd> ldltHs(Hs);
    if (ldltHs.info() == Eigen::Success) {
        x = ldltHs.solve(gs);
    } else {
        Eigen::JacobiSVD<Eigen::MatrixXd> svdHs(Hs, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd Sinv = svdHs.singularValues()
            .unaryExpr([](double s){ return (s > 1e-12) ? 1.0/s : 0.0; });
        x = svdHs.matrixV() * Sinv.asDiagonal() * svdHs.matrixU().transpose() * gs;
    }

    Eigen::Vector3d G = X.segment<3>(0);
    Eigen::Vector3d V = X.segment<3>(3);
    Eigen::Vector3d B = X.segment<3>(6);
    double normG = G.norm();
    if (normG > 0) {
        double scale = g_mag / normG;
        G *= scale;
        V *= scale;
        for(int i=0; i<N; ++i)
            for(int j=0; j<m; ++j)
                X(9 + i*m + j) *= scale;
    }

    solution.gravity  = G;
    solution.velocity = V;
    solution.bias     = B;

    for(auto &kv : mu_ij_map) {
        int i_idx = feat_to_idx[kv.first.first];
        int j_idx = img_to_idx[kv.first.second];
        solution.lambdas[{kv.first.first, kv.first.second}] =
            X(9 + i_idx*m + j_idx);
    }
    return true;
}
