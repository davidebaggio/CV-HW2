#ifndef DETECT_HPP
#define DETECT_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

struct card_corner
{
	cv::Rect window;
	std::string rank;
	double rankScore;
};

static float IoU(const cv::Rect &a, const cv::Rect &b);
std::vector<card_corner> non_max_suppression(std::vector<card_corner> &dets, float iouThresh = 0.3f);

void build_catalogue_sift(std::unordered_map<std::string, cv::Mat> &rankDesc);
void build_catalogue_tm(std::unordered_map<std::string, cv::Mat> &rankTemplate);

std::pair<std::string, int> detect_best(const cv::Mat &queryDesc, const std::unordered_map<std::string, cv::Mat> &catalog, cv::Ptr<cv::DescriptorMatcher> &matcher, float ratioThresh = 0.7f);
std::pair<std::string, double> best_template_match(const cv::Mat &patch, const std::unordered_map<std::string, cv::Mat> &templates, double thresh = 0.6);

std::string detect_card_sift(const cv::Mat &queryImg, std::unordered_map<std::string, cv::Mat> &rankDesc, int &rankScore, std::vector<cv::KeyPoint> &outQueryKpts);
std::vector<card_corner> detect_with_sliding_window(const cv::Mat &gray, std::unordered_map<std::string, cv::Mat> &rankTemplate, std::vector<double> scales = {1.0 /* , 0.8, 0.6 */}, cv::Size baseWindow = cv::Size(70, 125), int stride = 20, double rankThresh = 0.6);
std::vector<card_corner> detect_with_tl_window(const cv::Mat &gray, std::unordered_map<std::string, cv::Mat> &rankTemplate, std::vector<double> scales = {1.0}, cv::Size baseWindow = cv::Size(50, 170), double rankThresh = 0.9);

#endif // DETECT_HPP