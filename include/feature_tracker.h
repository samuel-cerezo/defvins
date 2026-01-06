/*
This file detect new points, does the optical flow and 
hold the tracks updated.
*/

#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "track_manager.h"

class FeatureTracker {
public:
    FeatureTracker();

    void processFrame(const cv::Mat& image, int frame_idx, TrackManager& track_manager);

private:
    cv::Mat prev_image_;
    std::vector<cv::Point2f> prev_points_;
    std::vector<int> prev_track_ids_;

    int max_corners_ = 500;
    double quality_level_ = 0.01;
    double min_distance_ = 10.0;
};
