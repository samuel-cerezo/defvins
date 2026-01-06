#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Detects and tracks features from img1 to img2 using KLT.
// Returns true if valid points were found and tracked.
bool detect_and_track_features(const std::string& img1_path,
                               const std::string& img2_path,
                               std::vector<cv::Point2f>& old_pts,
                               std::vector<cv::Point2f>& new_pts,
                               cv::Mat& img1,
                               cv::Mat& img2);
