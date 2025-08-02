// Zoren Martinez 2123873

#ifndef DETECT_HPP
#define DETECT_HPP

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

/**
 * @brief Loads a TorchScript model from file and sets it to evaluation mode.
 *
 * This function is responsible for loading a pre-trained card classifier model
 * in TorchScript format. If the model fails to load, the function throws an error.
 *
 * @param model_path Path to the `.pt` TorchScript model file.
 * @return Loaded TorchScript model ready for inference.
 */
torch::jit::script::Module load_card_model(const std::string &model_path);

/**
 * @brief Classifies a rank patch using a deep learning model.
 *
 * The input image is resized and normalized before being passed to the model.
 * The predicted class index is mapped to its corresponding rank label.
 *
 * @param value Grayscale image patch of the card rank area (assumed 1-channel).
 * @return Predicted rank label (e.g., "A", "10", "Q"), or "Unknown"/"Invalid" on error.
 */
std::string recognize_cards(const cv::Mat &value);

/**
 * @brief Extracts the rank symbol region using a center-based contour filtering method.
 *
 * This function assumes that the rank is located near the center of the patch and:
 * - Applies CLAHE for contrast enhancement.
 * - Binarizes the image to isolate foreground.
 * - Filters contours based on proximity to center and border exclusion.
 * - Optionally selects up to 2 central contours to build the mask.
 * - Pads the result on the left to preserve spatial structure.
 *
 * @param gray Grayscale input patch (typically from top-left of a card).
 * @return Cleaned and centered image patch of the detected rank area.
 */
cv::Mat extract_rank_patch_center_based(const cv::Mat &gray);

/**
 * @brief Adds a fixed-width white margin to the left side of the image.
 *
 * This is used to spatially shift the content to the right,
 * e.g., for improving CNN alignment or avoiding edge clipping.
 *
 * @param img Input grayscale image.
 * @param whiteCols Number of white columns to prepend.
 * @return New image with white padding on the left side.
 */
cv::Mat add_white_columns_left(const cv::Mat &img, int whiteCols);

#endif // DETECT_HPP
