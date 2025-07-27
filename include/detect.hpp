#ifndef DETECT_HPP
#define DETECT_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <torch/script.h>



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
std::vector<card_corner> detect_with_sliding_window(const cv::Mat &gray, std::unordered_map<std::string, cv::Mat> &rankTemplate, std::vector<double> scales = {1.0 /* , 0.8, 0.6 */}, cv::Size baseWindow = cv::Size(70, 125), int stride = 40, double rankThresh = 0.97);
std::vector<card_corner> detect_with_tl_window(const cv::Mat &gray, std::unordered_map<std::string, cv::Mat> &rankTemplate, std::vector<double> scales = {1.0}, cv::Size baseWindow = cv::Size(85, 115), double rankThresh = 0.97);
torch::jit::script::Module load_card_model(const std::string& model_path);
std::string recognize_cards(const cv::Mat &value);
cv::Mat extract_rank_patch(const cv::Mat& gray);
cv::Mat extract_rank_patch_center_based(const cv::Mat& gray);
cv::Mat addWhiteColumnsLeft(const cv::Mat& img, int whiteCols);

#endif // DETECT_HPP