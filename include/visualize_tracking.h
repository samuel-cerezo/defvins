#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

void visualize_tracking(const cv::Mat& img1,
                        const cv::Mat& img2,
                        const std::vector<cv::Point2f>& pts1,
                        const std::vector<cv::Point2f>& pts2,
                        const std::string& window_name = "Tracking",
                        bool save = false);
