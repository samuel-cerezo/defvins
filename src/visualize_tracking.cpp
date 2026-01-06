#include "visualize_tracking.h"
#include <opencv2/opencv.hpp>
#include <numeric>     
#include <random>       
#include <algorithm>    

void visualize_tracking(const cv::Mat& img1,
                        const cv::Mat& img2,
                        const std::vector<cv::Point2f>& pts1,
                        const std::vector<cv::Point2f>& pts2,
                        const std::string& window_name,
                        bool save) {
    cv::Mat img_vis;
    cv::Mat img1_color, img2_color;
    if (img1.channels() == 1)
        cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    else
        img1.copyTo(img1_color);

    if (img2.channels() == 1)
        cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
    else
        img2.copyTo(img2_color);

    cv::hconcat(img1_color, img2_color, img_vis);

    size_t N = std::min<size_t>(pts1.size(), pts2.size());
    size_t N_display = std::min<size_t>(20, N); 

    // índices aleatorios
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(42));  

    for (size_t i = 0; i < N_display; ++i) {
        int idx = indices[i];
        const auto& pt1 = pts1[idx];
        const auto& pt2 = pts2[idx] + cv::Point2f((float)img1.cols, 0);

        cv::circle(img_vis, pt1, 3, cv::Scalar(0, 0, 255), -1);  // rojo
        cv::circle(img_vis, pt2, 3, cv::Scalar(0, 255, 0), -1);  // verde
        cv::line(img_vis, pt1, pt2, cv::Scalar(255, 0, 0), 1);   // azul
    }
 
    cv::imwrite("matching.png", img_vis);
}
