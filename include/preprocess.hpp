#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include <opencv2/opencv.hpp>

void set_pixel_zero(cv::Mat &src, cv::Mat &dst, size_t col, int threshold);

void invert_pixel(cv::Mat &src, cv::Mat &dst);

void preprocessing(cv::Mat &image);

#endif // PREPROCESS_HPP