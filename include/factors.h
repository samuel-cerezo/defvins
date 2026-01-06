#pragma once
#include <ceres/ceres.h>
#include <Eigen/Dense>

/////////
struct PreintFactor {
  PreintFactor(const Eigen::Matrix3d& dR,
               const Eigen::Vector3d& dv,
               const Eigen::Vector3d& dp,
               double dt,
               const Eigen::Vector3d& g)
    : dR_(dR), dv_(dv), dp_(dp), dt_(dt), g_(g) {}

  template<typename T>
  bool operator()(const T* const qi, const T* const pi, const T* const vi,
                  const T* const qj, const T* const pj, const T* const vj,
                  const T* const g_param,   // si quieres estimar g, si no puedes fijarlo a g_
                  T* residuals) const
  {
    using Vec3 = Eigen::Matrix<T,3,1>;
    using Mat3 = Eigen::Matrix<T,3,3>;

    Eigen::Map<const Eigen::Quaternion<T>> Qi(qi);
    Eigen::Map<const Eigen::Quaternion<T>> Qj(qj);
    Mat3 Ri = Qi.normalized().toRotationMatrix();
    Mat3 Rj = Qj.normalized().toRotationMatrix();

    Eigen::Map<const Vec3> Pi(pi);
    Eigen::Map<const Vec3> Pj(pj);
    Eigen::Map<const Vec3> Vi(vi);
    Eigen::Map<const Vec3> Vj(vj);

    // gravedad: o bien viene como parámetro optimizable,
    // o puedes ignorar g_param y usar g_ fija.
    Eigen::Map<const Vec3> g_est(g_param);
    Vec3 g = g_est; // o Vec3 g = g_.cast<T>();

    // medidas preintegradas (constantes)
    Mat3 dR = dR_.cast<T>();
    Vec3 dv = dv_.cast<T>();
    Vec3 dp = dp_.cast<T>();
    T dt = T(dt_);

    // --- residual de rotación ---
    //Mat3 R_err = dR.transpose() * (Ri.transpose() * Rj);
    //Vec3 r_R = LogSO3(R_err);  // tu LogMap en SO(3)
    // --- residual de rotación ---
    //Mat3 R_err = dR.transpose() * (Ri.transpose() * Rj);
    //Vec3 r_R = lie::LogMap(R_err);  // 

    Mat3 E  = dR.transpose() * Ri.transpose() * Rj; // should be identity at optimum
    // Log map (approx): vee( E - E^T )/2 (válido cerca de I)
    Vec3 r_R;
    r_R << (E(2,1)-E(1,2))/T(2),
          (E(0,2)-E(2,0))/T(2),
          (E(1,0)-E(0,1))/T(2);
    
    // --- residual de velocidad ---
    Vec3 r_v = Ri.transpose() * (Vj - Vi - g * dt) - dv;

    // --- residual de posición ---
    Vec3 r_p = Ri.transpose() * (Pj - Pi - Vi * dt - T(0.5) * g * dt * dt) - dp;

    // empaquetar: [r_p; r_v; r_R] -> 9x1
    residuals[0] = r_p[0];
    residuals[1] = r_p[1];
    residuals[2] = r_p[2];

    residuals[3] = r_v[0];
    residuals[4] = r_v[1];
    residuals[5] = r_v[2];

    residuals[6] = r_R[0];
    residuals[7] = r_R[1];
    residuals[8] = r_R[2];

    return true;
  }

  Eigen::Matrix3d dR_;
  Eigen::Vector3d dv_;
  Eigen::Vector3d dp_;
  double dt_;
  Eigen::Vector3d g_;
};





/// Prior Gaussiano: || A (x - x0) - b ||^2  (compacto: implementa como residual lineal)
struct PriorFactor {
  PriorFactor(const Eigen::MatrixXd& A, const Eigen::VectorXd& b)
  : A_(A), b_(b) {}

  template<typename T>
  bool operator()(const T* const x, T* residuals) const {
    // x vectorizado: [R1(9) p1(3) v1(3) ... Rn pn vn bg(3) ba(3)]
    // Para empezar, puedes hacer A*vec(x) - b
    // (en producción conviene un ensamblado por bloques)
    Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,1>> xv(x, A_.cols());
    Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,1>>       r(residuals, A_.rows());
    r = A_.cast<T>() * xv - b_.cast<T>();
    return true;
  }

  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
};


struct GravityPriorCost {
    GravityPriorCost(const Eigen::Vector3d& g0, double w)
        : g0_(g0), w_(w) {}

    template<typename T>
    bool operator()(const T* const g, T* residuals) const {
        for (int i = 0; i < 3; i++)
            residuals[i] = T(w_) * (g[i] - T(g0_[i]));
        return true;
    }

    Eigen::Vector3d g0_;
    double w_;
};

struct ReprojectionCost {
  ReprojectionCost(const Eigen::Vector2d& uv_meas,
                   const Eigen::Matrix3d& K,
                   const Eigen::Matrix4d& T_body_cam)
    : uv_(uv_meas), K_(K)
  {
      R_bc_ = T_body_cam.topLeftCorner<3,3>();      // cam -> body
      t_bc_ = T_body_cam.topRightCorner<3,1>();     // en body
      R_cb_ = R_bc_.transpose();                    // body -> cam
  }

  template<typename T>
  bool operator()(const T* const q_B,   // KF body pose: R_wb
                  const T* const p_B,   // p_wb
                  const T* const X_W,   // punto en mundo (grafo ya llevado a mundo)
                  T* residuals) const 
  {
    using Vec3 = Eigen::Matrix<T,3,1>;
    using Mat3 = Eigen::Matrix<T,3,3>;

    Eigen::Map<const Eigen::Quaternion<T>> qB(q_B);
    Mat3 R_wb = qB.normalized().toRotationMatrix();   // body -> world
    Eigen::Map<const Vec3> p_wb(p_B);                 // body in world
    Eigen::Map<const Vec3> X_w(X_W);                  // punto en mundo

    // 1) Mundo -> cuerpo
    Vec3 X_b = R_wb.transpose() * (X_w - p_wb);

    // 2) Cuerpo -> cámara
    Mat3 R_cb = R_cb_.cast<T>();   // body -> cam
    Vec3 t_bc = t_bc_.cast<T>();   // cam origin in body frame
    Vec3 X_c  = R_cb * (X_b - t_bc);

    const T& Xc = X_c(0);
    const T& Yc = X_c(1);
    const T& Zc = X_c(2);

    T inv_Z = T(1.0) / Zc;
    T fx = T(K_(0,0));
    T fy = T(K_(1,1));
    T cx = T(K_(0,2));
    T cy = T(K_(1,2));

    T u = fx * Xc * inv_Z + cx;
    T v = fy * Yc * inv_Z + cy;

    residuals[0] = u - T(uv_(0));
    residuals[1] = v - T(uv_(1));
    return true;
  }

  Eigen::Vector2d uv_;
  Eigen::Matrix3d K_;
  Eigen::Matrix3d R_bc_, R_cb_;
  Eigen::Vector3d t_bc_;
};
