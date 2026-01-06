#pragma once
#include <map>
#include <vector>
#include <utility>
#include <Eigen/Dense>

/// Solución de inicialización/refinamiento 
struct VISolution {
  Eigen::Vector3d gravity;                   // G
  Eigen::Vector3d velocity;                  // V
  Eigen::Vector3d bias;                      // accelerometer bias (Ba)
  std::map<std::pair<int,int>, double> lambdas; // (i, jc) -> λ_ij (opcional)
};

namespace defvins {

/**
 * @brief Refinamiento no lineal con Ceres de la formulación cerrada (Martinelli-like).
 *
 * Requiere mapas compactos:
 *  - Gamma_comp: jc -> Γ_j (3x3)
 *  - S_comp:     jc -> S_j (3)
 *  - mu_comp: (i, jc) -> μ_ij (3), con densidad completa para i=1..N_eff y jc=1..C (jc=1 es referencia).
 *  - t_vals_comp: tamaño C, con t_vals_comp[0] = 0.0 (columna de referencia).
 *
 * @param N_eff   Número de puntos densos usados (i=1..N_eff)
 * @param C       Número de columnas compactas (jc=1..C)
 * @param g_mag   Norma objetivo de gravedad (p.ej. 9.81)
 * @param warm_start  Semilla opcional (G,V,B,λ); puede ser nullptr
 * @param out         Salida con (G,V,B) y λ opcional
 * @return true si la solución es usable (según Ceres), false en caso contrario
 */
bool SolveWithCeres(
  const std::map<int, Eigen::Matrix3d>& Gamma_comp,
  const std::map<int, Eigen::Vector3d>& S_comp,
  const std::map<std::pair<int,int>, Eigen::Vector3d>& mu_comp,
  const std::vector<double>& t_vals_comp,
  int N_eff, int C,
  double g_mag,
  const VISolution* warm_start, // puede ser nullptr
  VISolution& out
);

} // namespace defvins
