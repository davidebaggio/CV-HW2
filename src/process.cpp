#include "process.hpp"
#include "preprocess.hpp"

std::vector<std::vector<cv::Point>> filter_contours(const std::vector<std::vector<cv::Point>>& contours, double min_area, double min_perimeter)
{
    std::vector<std::vector<cv::Point>> filtered;

    // Loop over each contour in the input vector
    for (const auto& contour : contours)
    {
        // Compute the area of the current contour
        double area = cv::contourArea(contour);

        // Compute the perimeter (arc length) of the current contour
        double peri = cv::arcLength(contour, true);  // true indicates the contour is closed

        // Check if the contour satisfies both area and perimeter constraints
        if (area >= min_area && peri >= min_perimeter)
        {
            // If the contour meets the criteria, add it to the filtered results
            filtered.push_back(contour);
        }
    }

    // Return the filtered contours
    return filtered;
}



std::vector<cv::Point> find_closest_to_corners(const std::vector<cv::Point> &card, const cv::Size &img_size)
{
    // Define the top-right and bottom-left image corners
    cv::Point tr_corner(img_size.width - 1, 0);
    cv::Point bl_corner(0, img_size.height - 1);

    // Initialize the closest points to invalid positions
    cv::Point tr_closest(-1, -1), bl_closest(-1, -1);

    // Initialize the minimum distances with maximum possible values
    double tr_minDist = std::numeric_limits<double>::max();
    double bl_minDist = std::numeric_limits<double>::max();

    // Loop over all points in the card contour
    for (const auto &pt : card)
    {
        // Compute distance to top-right corner
        double tr_dist = cv::norm(pt - tr_corner);
        if (tr_dist < tr_minDist)
        {
            // Update the closest top-right point
            tr_minDist = tr_dist;
            tr_closest = pt;
        }

        // Compute distance to bottom-left corner
        double bl_dist = cv::norm(pt - bl_corner);
        if (bl_dist < bl_minDist)
        {
            // Update the closest bottom-left point
            bl_minDist = bl_dist;
            bl_closest = pt;
        }
    }

    // Return both closest points: [bottom-left, top-right]
    return {bl_closest, tr_closest};
}


void reorder_contour_with_bottom_left_first(std::vector<cv::Point> &contour, const cv::Point &bottom_left)
{
    // Find the iterator pointing to the bottom_left point in the contour
    auto it = std::find(contour.begin(), contour.end(), bottom_left);

    // If the point is found, rotate the vector so that this point becomes the first
    if (it != contour.end())
        std::rotate(contour.begin(), it, contour.end());
}


double point_line_distance(const cv::Point &p, const cv::Point &p1, const cv::Point &p2, double line_length)
{
    // Vector from p1 to p (d1) and from p1 to p2 (d2)
    cv::Point2f d1 = p - p1;
    cv::Point2f d2 = p2 - p1;

    // Cross product magnitude divided by line length gives perpendicular distance
    return std::abs(d1.x * d2.y - d1.y * d2.x) / line_length;
}

std::vector<int> find_local_maxima(const std::vector<double>& distances, int window_size, double min_prominence)
{
    std::vector<int> local_maxima;
    int n = static_cast<int>(distances.size());

    for (int i = window_size; i < n - window_size; ++i)
    {
        double current = distances[i];
        bool is_peak = true;

        // Check if current point is the max in the local window
        for (int j = i - window_size; j <= i + window_size; ++j)
        {
            if (distances[j] > current)
            {
                is_peak = false;
                break;
            }
        }

        if (is_peak)
        {
            // Apply prominence threshold (to skip flat/small bumps)
            double left = *std::min_element(distances.begin() + i - window_size, distances.begin() + i);
            double right = *std::min_element(distances.begin() + i + 1, distances.begin() + i + window_size + 1);
            double prominence = current - std::max(left, right);

            if (prominence >= min_prominence)
            {
                // Avoid nearby duplicate peaks
                if (local_maxima.empty() || (i - local_maxima.back()) > window_size / 2)
                    local_maxima.push_back(i);
            }
        }
    }

    return local_maxima;
}



std::vector<std::pair<int, int>> pair_indices_symmetric(const std::vector<int>& indices)
{
    std::vector<std::pair<int, int>> pairs;

    // Ensure the number of indices is even to form symmetric pairs
    if (indices.size() % 2 != 0)
    {
        std::cerr << "Errore: numero dispari di massimi locali (" << indices.size() << ")" << std::endl;
        return pairs;
    }

    int n = indices.size();

    // Pair i-th from the start with i-th from the end
    for (int i = 0; i < n / 2; ++i)
    {
        pairs.emplace_back(indices[i], indices[n - 1 - i]);
    }

    return pairs;
}

std::vector<std::vector<cv::Point>> extract_points_from_pairs(const std::vector<std::vector<cv::Point>>& cards, const std::vector<std::vector<cv::Point>>& ext_pts)
{
    std::vector<std::vector<cv::Point>> corner_pts;  // Result vector of corner sets per card
    cv::Point bl, br, tr, tl;                        // Bottom-left, bottom-right, top-right, top-left points
    
    std::vector<std::pair<int, int>> paired_indices; // Pairs of indices representing matching extrema
    std::vector<int> max_indices;                     // Indices of local maxima in distance array
    int i = 0, j = 0;                                 // Loop counters
    
    const int PIXEL_TOLERANCE = 7;                    // Pixel margin to adjust corner points for robustness
    
    std::vector<double> distances;                     // Stores perpendicular distances of contour points from reference line
    double line_length;                                // Length of reference line (between bottom-left and top-right)
    
    for (auto &card : cards)
    {
        bool first = true; // Flag to handle the first block separately
        
        // Adjust external reference points slightly by pixel tolerance for bottom-left and top-right corners
        bl = cv::Point(ext_pts[i][0].x - PIXEL_TOLERANCE, ext_pts[i][0].y + PIXEL_TOLERANCE);
        tr = cv::Point(ext_pts[i][1].x + PIXEL_TOLERANCE, ext_pts[i][1].y - PIXEL_TOLERANCE);

        i++; // Move to next card's external points
        
        // Compute the Euclidean distance (length) between bottom-left and top-right points
        line_length = cv::norm(tr - bl);
        
        distances.clear();
        distances.reserve(card.size());
        
        // For every point on the card contour, calculate its perpendicular distance from the line bl-tr
        for (const auto& pt : card)
            distances.push_back(point_line_distance(pt, bl, tr, line_length));
        
        // Find local maxima in the distance vector to detect key extremities (peaks)
        max_indices = find_local_maxima(distances, 25);
        
        // Pair the local maxima indices symmetrically from start and end
        paired_indices = pair_indices_symmetric(max_indices);
        
        std::cout << "Card " << i << ": " << paired_indices.size() << " coppie di estremi trovate." << std::endl;
        
        // Case 1: Only one pair of extrema found - likely a single quadrilateral block
        if (paired_indices.size() == 1)
        {

            br = cv::Point(card[paired_indices[0].first].x + PIXEL_TOLERANCE, card[paired_indices[0].first].y + PIXEL_TOLERANCE);    // Bottom-right point from first index of pair
            tl = cv::Point(card[paired_indices[0].second].x - PIXEL_TOLERANCE, card[paired_indices[0].second].y - PIXEL_TOLERANCE);  // Top-left point from second index of pair
            corner_pts.push_back({ bl, br, tr, tl });
        }
        // Case 2: Multiple pairs found - likely multiple blocks or subregions in the card contour
        else if (paired_indices.size() > 1)
        {
            for (j = 0; j < paired_indices.size() - 1; ++j)
            {
                // Adjust bottom-right and top-left points by pixel tolerance for robustness
                br = cv::Point(card[paired_indices[j].first].x + PIXEL_TOLERANCE, card[paired_indices[j].first].y + PIXEL_TOLERANCE);
                tl = cv::Point(card[paired_indices[j].second].x - PIXEL_TOLERANCE, card[paired_indices[j].second].y - PIXEL_TOLERANCE);
                
                if (first)
                {
                    // For the first block, use bl, br, adjusted point between br.x and tl.y, and tl
                    first = false;
                    corner_pts.push_back({ bl, br, cv::Point(br.x, tl.y), tl });
                }
                else
                {
                    // For subsequent blocks, create a quadrilateral from adjusted corner points
                    corner_pts.push_back({ cv::Point(tl.x, br.y), br, cv::Point(br.x, tl.y), tl });
                }
            }
            
            // Handle the last pair separately
            br = cv::Point(card[paired_indices[j].first].x + PIXEL_TOLERANCE, card[paired_indices[j].first].y + PIXEL_TOLERANCE);
            tl = cv::Point(card[paired_indices[j].second].x - PIXEL_TOLERANCE, card[paired_indices[j].second].y - PIXEL_TOLERANCE);
            
            // Final block uses the last tl, br, and the previously defined tr corner
            corner_pts.push_back({ cv::Point(tl.x, br.y), br, tr, tl });
        }
    }
    
    return corner_pts;  // Return all detected corner sets per card
}


std::vector<std::vector<cv::Point>> process(cv::Mat &image)
{
    std::vector<std::vector<cv::Point>> contours, cards, ext_points, lines;

    // Check if the input image is empty, return empty result if so
    if (image.empty())
    {
        std::cout << "Image is empty" << std::endl;
        return {};
    }

    // Find all external contours in the binary image
    // Use clone() to avoid modifying the original image data
    cv::findContours(image.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // Filter contours by minimum area and perimeter to retain likely card shapes
    cards = filter_contours(contours, 3000, 250);

    // For each detected card contour:
    for (auto &card : cards)
    {
        // Find points on the contour closest to the image's top-right and bottom-left corners
        auto corners = find_closest_to_corners(card, image.size());

        // Store these external corner points for further processing
        ext_points.push_back(corners);

        // Reorder the contour points so that the bottom-left point is first in the vector
        reorder_contour_with_bottom_left_first(card, corners[0]);
    }

    // Extract and refine corner points from the card contours and their external corners
    // Returns a vector of corner sets representing quadrilateral shapes of cards
    return extract_points_from_pairs(cards, ext_points);
}


void sharpen_image(cv::Mat &image)
{
    // Define a sharpening kernel that accentuates the center pixel 
    // and subtracts surrounding pixels to enhance edges
    cv::Mat sharpen = (cv::Mat_<float>(3, 3) << 
                        0, -1, 0,
                       -1,  5, -1,
                        0, -1, 0);
    
	// Apply the kernel to the image using filter2D in-place
    cv::filter2D(image, image, image.depth(), sharpen);
}


void sort_corners(std::vector<cv::Point2f> &pts)
{
    // Initialize corners with the first point
    cv::Point2f tl = pts[0], tr = pts[0], bl = pts[0], br = pts[0];
    for (auto &p : pts)
    {
        if (p.x + p.y < tl.x + tl.y)
            tl = p;
        if (p.x - p.y > tr.x - tr.y)
            tr = p;
        if (p.x - p.y < bl.x - bl.y)
            bl = p;
        if (p.x + p.y > br.x + br.y)
            br = p;
    }
    pts = {tl, tr, br, bl};
}


cv::Mat warp_to_rect(const cv::Mat &src, const std::vector<cv::Point> &quad, const cv::Size &dstSize)
{
    CV_Assert(quad.size() == 4);

    // Convert points to float
    std::vector<cv::Point2f> srcPts;
    srcPts.reserve(4);
    for (const auto &p : quad)
        srcPts.emplace_back(float(p.x), float(p.y));

    // Sort corners in top-left, top-right, bottom-right, bottom-left order
    sort_corners(srcPts);

    // Define destination points for the rectangular output
    std::vector<cv::Point2f> dstPts = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(float(dstSize.width - 1), 0.0f),
        cv::Point2f(float(dstSize.width - 1), float(dstSize.height - 1)),
        cv::Point2f(0.0f, float(dstSize.height - 1))
    };

    // Compute perspective transform matrix
    cv::Mat H = cv::getPerspectiveTransform(srcPts, dstPts);

    // Apply the perspective warp
    cv::Mat warped;
    cv::warpPerspective(src, warped, H, dstSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    return warped;
}




std::vector<cv::Mat> get_cards(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &rects)
{
    std::vector<cv::Mat> cards;

    cv::Size card_size(400, 600);
    for (auto &rect : rects)
    {
        // Warp card to a normalized rectangle
        cv::Mat card = warp_to_rect(src, rect, card_size);

        // Rotate 180 degrees for consistent upright orientation
        cv::Mat rot_matrix = cv::getRotationMatrix2D(cv::Point2f(card.cols / 2, card.rows / 2), 180, 1);
        cv::warpAffine(card, card, rot_matrix, cv::Size(card.cols, card.rows));

        // Preprocess (sharpen, binarize, etc.)
        preprocessing_card(card);


        //cv::imshow("warped", card);
        //cv::waitKey(0);

        cards.push_back(card);
    }

    return cards;
}
