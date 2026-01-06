#include "feature_tracker.h"
#include <iostream>
#include "visualize_tracking.h"

FeatureTracker::FeatureTracker() {}

void FeatureTracker::processFrame(const cv::Mat& image, int frame_idx, TrackManager& track_manager) {
    if (prev_image_.empty()) {
        // First frame: detect initial features
        std::vector<cv::Point2f> keypoints;
        double min_distance_ = 40;
        cv::goodFeaturesToTrack(image, keypoints, 100, quality_level_, min_distance_);  // antes era 20

        for (const auto& kp : keypoints) {
            int id = track_manager.createTrack(frame_idx, Eigen::Vector2d(kp.x, kp.y));
            prev_track_ids_.push_back(id);
        }

        prev_points_ = keypoints;
        prev_image_ = image.clone();
        return;
    }

    // Track points from previous image
    std::vector<cv::Point2f> next_points;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_image_, image, prev_points_, next_points, status, err);


    std::vector<cv::Point2f> new_points;
    std::vector<int> new_track_ids;

    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) continue;
        const auto& pt = next_points[i];
        int track_id = prev_track_ids_[i];

        // Agregar observación al track correspondiente
        track_manager.addObservation(track_id, frame_idx, Eigen::Vector2d(pt.x, pt.y));

        new_points.push_back(pt);
        new_track_ids.push_back(track_id);
    }

    // Detectar nuevos puntos si el número bajó
    if (new_points.size() < 0.8 * 100) {
        std::vector<cv::Point2f> detected;
        int num_to_add = std::min(10, max_corners_ - (int)new_points.size());
        cv::goodFeaturesToTrack(image, detected, num_to_add, quality_level_, min_distance_);
        
        for (const auto& kp : detected) {
            int id = track_manager.createTrack(frame_idx, Eigen::Vector2d(kp.x, kp.y));
            new_points.push_back(kp);
            new_track_ids.push_back(id);
        }
    }

    prev_image_ = image.clone();
    prev_points_ = new_points;
    prev_track_ids_ = new_track_ids;

}
