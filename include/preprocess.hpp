// Francesco Pivotto 2158296

#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include <opencv2/opencv.hpp>

/**
 * @brief Enhances a card image to make text and edges more legible for OCR.
 *
 * This function converts the input image to grayscale, applies CLAHE to improve
 * local contrast, sharpens edges to accentuate features, applies Gaussian blur
 * to reduce noise, and finally binarizes the image using Otsu's threshold.
 *
 * @param image Input/output cv::Mat representing the card image.
 * Must be a valid BGR image initially. After processing, it becomes grayscale binary.
 */
void preprocessing_card(cv::Mat &image);

/**
 * @brief Isolates bright, low-saturation regions (e.g. light beige or white areas)
 * within an image using an HSV-based mask and morphological operations.
 *
 * This function converts the input BGR image to HSV, thresholds to detect
 * "white-like" areas, applies dilation to close gaps, finds and fills external
 * contours, and then erodes to smooth shapes. The resulting image is binary,
 * where white-like regions are white (255) and the rest is black (0).
 *
 * @param image Input/output cv::Mat representing the original image.
 * Must be a valid BGR image initially. After processing, it becomes grayscale binary.
 */
void preprocessing_image(cv::Mat &image);

#endif // PREPROCESS_HPP