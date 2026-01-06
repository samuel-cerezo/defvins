#include "ceres_refine.h"
#include <ceres/ceres.h>
#include <thread>

namespace { // ---------- Cost functors ----------

struct EqJ_FirstPointResidual {
  EqJ_FirstPointResidual(double tj,
                         const Eigen::Matrix3d& Gamma_j,
                         const Eigen::Vector3d& S_j,
                         const Eigen::Vector3d& mu00,
                         const Eigen::Vector3d& mu0j)
  : tj_(tj), Gamma_j_(Gamma_j), S_j_(S_j), mu00_(mu00), mu0j_(mu0j) {}

  template<typename T>
  bool operator()(const T* const G,  // 3
                  const T* const V,  // 3
                  const T* const B,  // 3
                  const T* const lam00, // 1
                  const T* const lam0j, // 1
                  T* residuals) const
  {
    using Vec3T = Eigen::Matrix<T,3,1>;
    using Mat3T = Eigen::Matrix<T,3,3>;

    Eigen::Map<const Vec3T> Gv(G);
    Eigen::Map<const Vec3T> Vv(V);
    Eigen::Map<const Vec3T> Bv(B);

    const T tj = T(tj_);
    Vec3T r = (-T(0.5)*tj*tj)*Gv
            + (-tj)*Vv
            + (Gamma_j_.template cast<T>() * Bv)
            + lam00[0]*mu00_.template cast<T>()
            - lam0j[0]*mu0j_.template cast<T>()
            - S_j_.template cast<T>();

    residuals[0] = r(0); residuals[1] = r(1); residuals[2] = r(2);
    return true;
  }

  double tj_;
  Eigen::Matrix3d Gamma_j_;
  Eigen::Vector3d S_j_, mu00_, mu0j_;
};

struct EqIJ_CouplingResidual {
  EqIJ_CouplingResidual(const Eigen::Vector3d& mu00,
                        const Eigen::Vector3d& mu0j,
                        const Eigen::Vector3d& mui0,
                        const Eigen::Vector3d& muij)
  : mu00_(mu00), mu0j_(mu0j), mui0_(mui0), muij_(muij) {}

  template<typename T>
  bool operator()(const T* const lam00, // 1
                  const T* const lam0j, // 1
                  const T* const lami0, // 1
                  const T* const lamij, // 1
                  T* residuals) const
  {
    using Vec3T = Eigen::Matrix<T,3,1>;
    Vec3T r = lam00[0]*mu00_.template cast<T>()
            - lam0j[0]*mu0j_.template cast<T>()
            - lami0[0]*mui0_.template cast<T>()
            + lamij[0]*muij_.template cast<T>();

    residuals[0] = r(0); residuals[1] = r(1); residuals[2] = r(2);
    return true;
  }

  Eigen::Vector3d mu00_, mu0j_, mui0_, muij_;
};

struct GravityNormPenalty {
  explicit GravityNormPenalty(double g_mag) : g_mag_(g_mag) {}
  template<typename T>
  bool operator()(const T* const G, T* residuals) const {
    const T n = ceres::sqrt(G[0]*G[0] + G[1]*G[1] + G[2]*G[2]);
    residuals[0] = n - T(g_mag_);
    return true;
  }
  double g_mag_;
};

struct BiasL2Penalty {
  explicit BiasL2Penalty(double sigma) : inv_sigma_(1.0/sigma) {}
  template<typename T>
  bool operator()(const T* const B, T* residuals) const {
    residuals[0] = T(inv_sigma_) * B[0];
    residuals[1] = T(inv_sigma_) * B[1];
    residuals[2] = T(inv_sigma_) * B[2];
    return true;
  }
  double inv_sigma_;
};

} // namespace

// ---------- SolveWithCeres ----------
namespace defvins {

bool SolveWithCeres(
  const std::map<int, Eigen::Matrix3d>& Gamma_comp,
  const std::map<int, Eigen::Vector3d>& S_comp,
  const std::map<std::pair<int,int>, Eigen::Vector3d>& mu_comp,
  const std::vector<double>& t_vals_comp,
  int N_eff, int C,
  double g_mag,
  const VISolution* warm_start,
  VISolution& out
)
{
  struct Blocks {
    double G[3]{}, V[3]{}, B[3]{};
    std::map<std::pair<int,int>, double> lambda; // (i,jc) -> λ_ij
  } x;

  // Warm-start si existe
  if (warm_start) {
    x.G[0]=warm_start->gravity.x(); x.G[1]=warm_start->gravity.y(); x.G[2]=warm_start->gravity.z();
    x.V[0]=warm_start->velocity.x(); x.V[1]=warm_start->velocity.y(); x.V[2]=warm_start->velocity.z();
    x.B[0]=warm_start->bias.x();     x.B[1]=warm_start->bias.y();     x.B[2]=warm_start->bias.z();
    // (λ opcional: podrías inicializarlas con warm_start->lambdas si las guardas)
  }

  // Inicializar lambdas a 0.0
  for (int i=1;i<=N_eff;++i)
    for (int jc=1;jc<=C;++jc)
      x.lambda[{i,jc}] = 0.0;

  ceres::Problem problem;

  // Parámetros G, V, B
  problem.AddParameterBlock(x.G, 3);
  problem.AddParameterBlock(x.V, 3);
  problem.AddParameterBlock(x.B, 3);

  // Parámetros λ_ij
  for (auto& kv : x.lambda) {
    problem.AddParameterBlock(&kv.second, 1);
  }

  const int i0 = 1; // punto de referencia

  // Ecuaciones con S_j (jc=2..C)
  for (int jc=2; jc<=C; ++jc) {
    const double tj = t_vals_comp[jc-1]; // t[0]=0.0
    const auto& Gamma_j = Gamma_comp.at(jc);
    const auto& S_j     = S_comp.at(jc);
    const auto& mu00    = mu_comp.at({i0,1});
    const auto& mu0j    = mu_comp.at({i0,jc});

    auto* cf = new ceres::AutoDiffCostFunction<EqJ_FirstPointResidual, 3, 3,3,3, 1,1>(
                 new EqJ_FirstPointResidual(tj, Gamma_j, S_j, mu00, mu0j));
    ceres::LossFunction* huber = new ceres::HuberLoss(1.0);
    problem.AddResidualBlock(cf, huber,
                             x.G, x.V, x.B,
                             &x.lambda[{i0,1}], &x.lambda[{i0,jc}]);
  }

  // Ecuaciones de acoplamiento (i>=2, jc>=2)
  for (int i=2; i<=N_eff; ++i) {
    const auto& mui0 = mu_comp.at({i,1});
    for (int jc=2; jc<=C; ++jc) {
      const auto& muij = mu_comp.at({i,jc});
      const auto& mu00 = mu_comp.at({i0,1});
      const auto& mu0j = mu_comp.at({i0,jc});

      auto* cf = new ceres::AutoDiffCostFunction<EqIJ_CouplingResidual, 3, 1,1,1,1>(
                   new EqIJ_CouplingResidual(mu00, mu0j, mui0, muij));
      ceres::LossFunction* huber = new ceres::HuberLoss(1.0);
      problem.AddResidualBlock(cf, huber,
                               &x.lambda[{i0,1}], &x.lambda[{i0,jc}],
                               &x.lambda[{i,1}],  &x.lambda[{i,jc}]);
    }
  }

  // Penalizaciones
  {
    auto* cf = new ceres::AutoDiffCostFunction<GravityNormPenalty, 1, 3>(
                 new GravityNormPenalty(g_mag));
    problem.AddResidualBlock(cf, nullptr, x.G);
  }
  {
    auto* cf = new ceres::AutoDiffCostFunction<BiasL2Penalty, 3, 3>(
                 new BiasL2Penalty(/*sigma=*/0.5));
    problem.AddResidualBlock(cf, nullptr, x.B);
  }

  // Schur: (G,V,B) como "cámara", λ como "puntos"
  //ceres::ParameterBlockOrdering ordering;
  //ordering.AddElementToGroup(x.G, 0);
  //ordering.AddElementToGroup(x.V, 0);
  //ordering.AddElementToGroup(x.B, 0);
  //for (auto& kv : x.lambda) ordering.AddElementToGroup(&kv.second, 1);

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::SPARSE_SCHUR;          // o DENSE_SCHUR si N,C pequeños
  opts.use_explicit_schur_complement = true;
  //opts.parameter_block_ordering = &ordering;
  opts.max_num_iterations = 50;
  opts.function_tolerance = 1e-10;
  opts.gradient_tolerance = 1e-12;
  opts.num_threads = std::max(1u, std::thread::hardware_concurrency());

  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);
  // std::cout << summary.FullReport() << "\n";

  // Salida
  out.gravity  = Eigen::Vector3d(x.G[0], x.G[1], x.G[2]);
  out.velocity = Eigen::Vector3d(x.V[0], x.V[1], x.V[2]);
  out.bias     = Eigen::Vector3d(x.B[0], x.B[1], x.B[2]);
  // out.lambdas = x.lambda; // descomenta si quieres devolverlas

  return summary.IsSolutionUsable();
}

} // namespace defvins
