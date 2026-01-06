#include "feature_tracking.h"
#include <opencv2/opencv.hpp>

bool detect_and_track_features(const std::string& img1_path,
                               const std::string& img2_path,
                               std::vector<cv::Point2f>& old_pts,
                               std::vector<cv::Point2f>& new_pts,
                               cv::Mat& img1,
                               cv::Mat& img2) {
    img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) return false;

    std::vector<cv::Point2f> points1;
    cv::goodFeaturesToTrack(img1, points1, 1000, 0.01, 10);
    if (points1.empty()) return false;

    std::vector<uchar> status;
    std::vector<float> err;
    
    cv::Size winSize(21, 21);
    int maxLevel = 3;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img1, img2, points1, new_pts, status, err, winSize, maxLevel, criteria);

    // Filter valid tracked points
    old_pts.clear();
    std::vector<cv::Point2f> filtered_new;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            old_pts.push_back(points1[i]);
            filtered_new.push_back(new_pts[i]);
        }
    }
    new_pts = filtered_new;
    return !old_pts.empty();
}
