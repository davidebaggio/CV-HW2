#ifndef PROCESS_HPP
#define PROCESS_HPP

#include <opencv2/opencv.hpp>

/**
 * @brief Filters contours based on area and perimeter thresholds.
 *
 * @param contours Input vector of contours to be filtered.
 * @param min_area Minimum contour area to keep.
 * @param min_perimeter Minimum perimeter length to keep.
 * @return Vector of contours that satisfy both area and perimeter constraints.
 */
std::vector<std::vector<cv::Point>> filter_contours(const std::vector<std::vector<cv::Point>> &contours, double min_area, double min_perimeter);

/**
 * @brief Finds the points in the contour closest to the top-right and bottom-left
 * corners of the image.
 *
 * @param card A single contour representing a detected card.
 * @param img_size Size of the original image to define corner references.
 * @return A pair of points {bottom-left, top-right} closest to image corners.
 */
std::vector<cv::Point> find_closest_to_corners(const std::vector<cv::Point> &card, const cv::Size &img_size);

/**
 * @brief Reorders the contour so that the bottom-left point comes first.
 *
 * This is useful for consistent orientation of contours in later processing steps.
 *
 * @param contour The contour to reorder.
 * @param bottom_left The point to move to the start of the contour.
 */
void reorder_contour_with_bottom_left_first(std::vector<cv::Point> &contour, const cv::Point &bottom_left);

/**
 * @brief Computes the perpendicular distance from a point to a line defined by two points.
 *
 * This function uses the cross product method to calculate the shortest distance between
 * a point `p` and a line segment from `p1` to `p2`. It assumes you already know the length
 * of the line (passed as `line_length`) to avoid recalculating it if this function is used repeatedly.
 *
 * @param p            The point from which the distance is calculated.
 * @param p1           One endpoint of the line.
 * @param p2           The other endpoint of the line.
 * @param line_length  Precomputed length of the line segment (norm of p2 - p1).
 * @return             The perpendicular distance from point p to the line defined by p1 and p2.
 */
double point_line_distance(const cv::Point &p, const cv::Point &p1, const cv::Point &p2, double line_length);

/**
 * @brief Identifies local maxima in a 1D array using a moving window.
 *
 * This function scans through a vector of values and determines which elements
 * are local maxima within a specified window range. An element is considered a
 * local maximum if it is not smaller than any of its neighbors within the window.
 *
 * @param distances    The vector of values in which to find local peaks.
 * @param window_size  Half-size of the window used to compare neighboring elements.
 * @return             A vector of indices corresponding to the local maxima found.
 */
std::vector<int> find_local_maxima(const std::vector<double> &distances, int window_size, double min_prominence = 10.0);

/**
 * @brief Pairs indices symmetrically from the start and end of a sorted list.
 *
 * This function is useful in geometric contexts, such as pairing feature points
 * detected symmetrically around a central axis or line. It assumes the input vector
 * is already sorted or ordered in a meaningful way.
 *
 * Example: input = {2, 5, 9, 13} → output = {{2,13}, {5,9}}
 *
 * @param indices A vector of integer indices to pair.
 * @return A vector of index pairs. If the input size is odd, returns an empty vector and logs an error.
 */
std::vector<std::pair<int, int>> pair_indices_symmetric(const std::vector<int> &indices);

/**
 * @brief Extracts corner points from contour point pairs based on external extrema.
 *
 * This function processes a list of contours (typically representing cards or blocks in an image)
 * and, using externally provided reference points, computes the corner points that define
 * rectangular regions (e.g., cells or blocks) within each contour.
 *
 * The method works by:
 * - Calculating the perpendicular distance of each contour point from the line connecting the external corners,
 * - Identifying local maxima in the distance profile to detect significant internal extremities,
 * - Pairing these extremities symmetrically to define sub-blocks,
 * - Building sets of four corner points for each detected region.
 *
 * If only one pair of extrema is found, the contour is assumed to represent a single block.
 * If multiple pairs are found, the contour is assumed to contain sub-blocks (e.g., multiple rows or cells).
 *
 * @param cards    A vector of contours (each a vector of cv::Point), representing the shapes of cards or regions.
 * @param ext_pts  A vector of external reference points for each contour. Each element must contain two points:
 *                 [0] is the bottom-left (bl) corner, [1] is the top-right (tr) corner.
 *
 * @return A vector of quadrilaterals (each a vector of 4 cv::Point elements), representing the detected regions
 *         in each contour. The order of points for each quadrilateral is:
 *         bottom-left, bottom-right, top-right, top-left.
 *
 * @note
 * - Corner points are slightly adjusted with a pixel tolerance to improve robustness.
 * - This function relies on helper functions `point_line_distance`, `find_local_maxima`, and
 *   `pair_indices_symmetric`, which must be defined elsewhere.
 * - Debug information is printed to `std::cout`, indicating the number of extrema pairs found per contour.
 */
std::vector<std::vector<cv::Point>> extract_points_from_pairs(const std::vector<std::vector<cv::Point>> &cards, const std::vector<std::vector<cv::Point>> &ext_pts);

/**
 * @brief Detects and extracts quadrilateral regions (e.g., cards) from a binary image.
 *
 * This function performs a complete processing pipeline to detect rectangular objects (such as cards)
 * from a binary image, typically pre-processed (e.g., thresholded or edge-detected).
 * It identifies external contours, filters them based on geometric constraints, and refines them
 * by locating corner points and reordering contour data for consistent downstream processing.
 *
 * The method involves the following steps:
 * - Detect external contours in the binary image using OpenCV's `findContours`.
 * - Filter out contours that do not meet minimum area and perimeter thresholds to retain only likely card shapes.
 * - For each valid contour:
 *    - Determine the two contour points closest to the image's bottom-left and top-right corners.
 *    - Save these external points as references for further corner refinement.
 *    - Reorder the contour points such that the bottom-left corner appears first.
 * - Optionally, visualize all detected contours for debugging or inspection purposes.
 * - Extract and return a set of ordered corner points representing rectangular regions for each card.
 *
 * @param image  Input image (`cv::Mat`), typically a binary (black and white) image.
 *               Must not be empty. The image is not modified directly.
 *
 * @return A vector of quadrilateral shapes (`std::vector<std::vector<cv::Point>>`),
 *         where each inner vector contains four points ordered as:
 *         bottom-left, bottom-right, top-right, top-left.
 *
 */
std::vector<std::vector<cv::Point>> process(cv::Mat &image);

/**
 * @brief Sharpens the input image using a simple 3x3 convolution kernel.
 *
 * This operation enhances edges and fine details by emphasizing pixel intensity differences.
 * Commonly used as a preprocessing step before tasks like OCR, contour detection, or feature extraction.
 *
 * @param image Input/output image that will be modified in place.
 */
void sharpen_image(cv::Mat &image);

/**
 * @brief Orders a set of 4 points in the sequence: top-left, top-right, bottom-right, bottom-left.
 *
 * This ordering is required for consistent perspective transformations (homography).
 * The method uses the sums and differences of point coordinates:
 * - Top-left has the smallest sum (x + y).
 * - Bottom-right has the largest sum (x + y).
 * - Top-right has the largest difference (x - y).
 * - Bottom-left has the smallest difference (x - y).
 *
 * @param pts Vector of 4 points to be sorted in place.
 */
void sort_corners(std::vector<cv::Point2f> &pts);

/**
 * @brief Applies a perspective transform to a quadrilateral region, warping it into a rectangle.
 *
 * This function rectifies the specified quadrilateral region from the source image into a
 * rectangular image of the given size. It is typically used for card or document extraction
 * to normalize the perspective before further processing.
 *
 * @param src The input source image.
 * @param quad Vector of 4 points representing the corners of the quadrilateral region in the source image.
 *             The points do not need to be ordered; the function will order them internally.
 * @param dstSize The size (width, height) of the output warped image (rectangular).
 * @return The warped image containing the perspective-corrected rectangle.
 */
cv::Mat warp_to_rect(const cv::Mat &src, const std::vector<cv::Point> &quad, const cv::Size &dstSize);

/**
 * @brief Warps and preprocesses card regions extracted from the source image.
 *
 * For each detected card-like quadrilateral contour, this function:
 * - Applies a perspective warp to obtain a front-facing rectangular image of fixed size.
 * - Rotates the card image by 180 degrees to ensure consistent orientation.
 * - Applies preprocessing (e.g., contrast enhancement, binarization) to prepare for OCR or further analysis.
 *
 * @param src The original full input image.
 * @param rects Vector of 4-point contours representing detected cards (quadrilaterals).
 * @return A vector of preprocessed card images, each warped and oriented consistently.
 */
std::vector<cv::Mat> get_cards(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &rects);

#endif // PROCESS_HPP