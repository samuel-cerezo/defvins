#include "imu_preint.h"
#include <algorithm>

using M3 = Eigen::Matrix3d;
using V3 = Eigen::Vector3d;

static inline M3 ExpSO3(const V3& wdt){
  double th = wdt.norm();
  M3 I = M3::Identity();
  if (th < 1e-12) return I;
  V3 a = wdt / th;
  M3 K; K <<     0,-a.z(), a.y(),
               a.z(),    0,-a.x(),
              -a.y(), a.x(),    0;
  return I + std::sin(th)*K + (1-std::cos(th))*(K*K);
}

PreintMeas Preintegrate(const std::vector<ImuSample>& imu, double t_i, double t_j,
                        const V3& bg, const V3& ba, const V3& g)
{
  PreintMeas out{};
  out.dR.setIdentity();
  out.dv.setZero();
  out.dp.setZero();
  out.dt = t_j - t_i;
  if (out.dt <= 0) return out;

  // recolectar segmento [t_i, t_j]
  std::vector<ImuSample> seg;
  seg.reserve(128);
  for (size_t k=1;k<imu.size();++k){
    if (imu[k-1].t >= t_i && imu[k].t <= t_j) seg.push_back(imu[k-1]);
  }
  if (seg.size() < 1) return out;

  double t = t_i;
  M3 R = M3::Identity();
  V3 v = V3::Zero();
  V3 p = V3::Zero();

  for (size_t k=0;k<seg.size();++k){
    double dt = (k+1<seg.size()) ? (seg[k+1].t - seg[k].t) : (t_j - seg.back().t);
    if (dt <= 0) continue;
    V3 w = seg[k].omega - bg;
    V3 a = seg[k].acc   - ba;

    M3 dR = ExpSO3(w*dt);
    R = R * dR;
    v += (R * a + g) * dt;      
    p += v * dt + 0.5 * (R * a + g) * dt * dt;
    t += dt;
  }

  out.dR = R;
  out.dv = v;
  out.dp = p;
  return out;
}
