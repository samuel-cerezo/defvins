#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <pybind11/eigen.h> 
#include "c2p_wrapper.h"

namespace py = pybind11;

C2PResult run_c2p(const std::vector<Eigen::Vector3d>& f0_inliers,
             const std::vector<Eigen::Vector3d>& f1_inliers,
             int frame_idx)
{
    
   
    // Add your script directory to sys.path
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("append")("/Users/samucerezo/dev/src/Vi-init-cpp/scripts");

    // Import your wrapper module
    py::module_ wrapper;

    try {
        wrapper = py::module_::import("nonmin_pose_wrapper");
        //std::cout << "[INFO] Módulo 'nonmin_pose_wrapper' importado correctamente." << std::endl;

        if (!py::hasattr(wrapper, "c2p_")) {
            throw std::runtime_error("The module does not have the function 'c2p_'");
        }
        //std::cout << "[INFO] Función 'c2p_' encontrada en el módulo." << std::endl;

    } catch (const py::error_already_set& e) {
        std::cerr << "[ERROR importing 'c2p_':\n" << e.what() << std::endl;
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero(), false, false};
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero(), false, false};
    }


    // Convert f0 and f1 to numpy arrays (shape: 3xN)
    size_t N = f0_inliers.size();

    if (N == 0) {
        std::cerr << "[Frame " << frame_idx << "] run_c2p: No inliers, skipping.\n";
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero(), false, false};

    }

    if (f0_inliers.empty() || f1_inliers.empty()) {
        std::cerr << "[Frame " << frame_idx << "] Error: f0_inliers or f1_inliers is empty." << std::endl;
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero(), false, false};

    }

    if (f0_inliers.size() != f1_inliers.size()) {
        std::cerr << "[Frame " << frame_idx << "] Error:number of inliers does not match: f0=" 
                << f0_inliers.size() << ", f1=" << f1_inliers.size() << std::endl;
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero(), false, false};

    }

    //std::cout << "[Frame " << frame_idx << "] Número de inliers: " << f0_inliers.size() << std::endl;



    Eigen::MatrixXd f0_mat(3, N), f1_mat(3, N);
    for (size_t i = 0; i < N; ++i) {
        f0_mat.col(i) = f0_inliers[i];
        f1_mat.col(i) = f1_inliers[i];
    }

    py::array_t<double> f0_np = py::cast(f0_mat);
    py::array_t<double> f1_np = py::cast(f1_mat);

    try {
        
        auto f0_shape = f0_np.shape();
        auto f1_shape = f1_np.shape();
        auto result_obj = wrapper.attr("c2p_")(f0_np, f1_np);

        if (!py::isinstance<py::tuple>(result_obj)) {
            throw std::runtime_error("Expected a tuple return from c2p_");
        }

        auto result = result_obj.cast<py::tuple>();
        

        auto E_np = result[0].cast<py::array_t<double>>();
        auto R_np = result[1].cast<py::array_t<double>>();
        auto t_np = result[2].cast<py::array_t<double>>();
        bool is_optimal = result[3].cast<bool>();
        bool is_pure_rot = result[4].cast<bool>();

        Eigen::Matrix3d E = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(E_np.data());
        Eigen::Matrix3d R = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_np.data());
        Eigen::Vector3d t = Eigen::Map<const Eigen::Vector3d>(t_np.data());


        return {E, R, t, is_optimal, is_pure_rot};

    } catch (const py::error_already_set& e) {
        std::cerr << "[Frame " << frame_idx << "] Python error: " << e.what() << std::endl;
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero(), false, false};

    } catch (const std::exception& e) {
        std::cerr << "[Frame " << frame_idx << "] Error en run_c2p: " << e.what() << std::endl;
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero(), false, false};

    }

}
