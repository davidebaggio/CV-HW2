#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "json.hpp"

/**
 * @brief Computes the Intersection over Union (IoU) between two polygons.
 *
 * This function calculates the IoU metric by first computing the bounding rectangles
 * for the predicted and ground truth polygons, then measuring the area of their
 * intersection and union. The IoU is used to quantify the overlap between the two shapes.
 *
 * @param pred Vector of cv::Point representing the predicted polygon.
 * @param gt Vector of cv::Point representing the ground truth polygon.
 * @return A double in the range [0.0, 1.0] representing the IoU value.
 */
double compute_iou(const std::vector<cv::Point> &pred, const std::vector<cv::Point> &gt);

/**
 * @brief Evaluates predictions against ground truth annotations from a COCO-style JSON file.
 *
 * This function reads the ground truth annotations from a JSON file, matches predictions
 * to ground truth polygons using IoU and class label comparison, and computes precision,
 * recall, and F1-score metrics based on the matches.
 *
 * @param json_path Path to the COCO-format JSON file containing ground truth annotations.
 * @param predictions Map from image filename to a vector of predicted polygons and their labels.
 * @param iou_threshold Minimum IoU required to consider a prediction as correct.
 */
void evaluate_predictions(const std::string &json_path,
						  const std::map<std::string, std::vector<std::pair<std::vector<cv::Point>, std::string>>> &predictions,
						  double iou_threshold = 0.5);

/**
 * @brief Returns the Hi-Lo card counting value for a given card rank.
 *
 * Based on the Hi-Lo system commonly used in blackjack card counting, this function
 * assigns a value to a card rank: low cards (2–6) are +1, neutral cards (7–9) are 0,
 * and high cards (10–A) are -1.
 *
 * @param card_value String representing the card rank (e.g., "2", "10", "K", "A").
 * @return An integer representing the Hi-Lo count value: +1, 0, or -1.
 */
int get_hi_lo_value(const std::string &card_value);

/**
 * @brief Returns a color corresponding to a Hi-Lo card count value.
 *
 * This function maps Hi-Lo count values to BGR color codes for visualization:
 * positive values map to green, negative to red, and zero to gray.
 *
 * @param value Integer count value from the Hi-Lo system.
 * @return cv::Scalar representing the corresponding BGR color.
 */
cv::Scalar get_color_for_value(int value);

#endif