#ifndef PROCESS_HPP
#define PROCESS_HPP

#include <opencv2/opencv.hpp>

std::vector<std::vector<cv::Point>> process(cv::Mat &image);

void sharpen_image(cv::Mat &image);

void sort_corners(std::vector<cv::Point2f> &pts);

cv::Mat warp_to_rect(const cv::Mat &src, const std::vector<cv::Point> &quad, const cv::Size &dstSize);

std::vector<cv::Mat> get_cards(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &rects);

#endif // PROCESS_HPP