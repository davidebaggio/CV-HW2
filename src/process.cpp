#include "process.hpp"
#include "preprocess.hpp"


std::vector<std::vector<cv::Point>> filter_contours(const std::vector<std::vector<cv::Point>>& contours, double min_area, double min_perimeter)
{
	std::vector<std::vector<cv::Point>> filtered;
	for (const auto& contour : contours)
	{
		double area = cv::contourArea(contour);
		double peri = cv::arcLength(contour, true);

		if (area >= min_area && peri >= min_perimeter)
		{
			filtered.push_back(contour);
		}
	}
	return filtered;
}

std::vector<cv::Point> find_closest_to_corners(const std::vector<cv::Point> &card, const cv::Size &img_size)
{
    cv::Point tr_corner(img_size.width - 1, 0), bl_corner(0, img_size.height - 1);
    cv::Point tr_closest(-1, -1), bl_closest(-1, -1);

    double tr_minDist = std::numeric_limits<double>::max();
    double bl_minDist = std::numeric_limits<double>::max();

	for (const auto &pt : card)
	{
		double tr_dist = cv::norm(pt - tr_corner);
		if (tr_dist < tr_minDist)
		{
			tr_minDist = tr_dist;
			tr_closest = pt;
		}

		double bl_dist = cv::norm(pt - bl_corner);
		if (bl_dist < bl_minDist)
		{
			bl_minDist = bl_dist;
			bl_closest = pt;
		}
	}
    

    return {bl_closest, tr_closest};
}

void reorder_contour_with_bottom_left_first(std::vector<cv::Point> &contour, const cv::Point &bottom_left)
{
    auto it = std::find(contour.begin(), contour.end(), bottom_left);
    if (it != contour.end())
        std::rotate(contour.begin(), it, contour.end());
    
}


double point_line_distance(const cv::Point &p, const cv::Point &p1, const cv::Point &p2, double line_length)
{
    cv::Point2f d1 = p - p1;
    cv::Point2f d2 = p2 - p1;
    return std::abs(d1.x * d2.y - d1.y * d2.x) / line_length;
}


std::vector<int> find_local_maxima(const std::vector<double>& distances, int window_size)
{
    std::vector<int> local_maxima;
    int n = distances.size();
    for (int i = 0; i < n; ++i)
    {
        bool is_max = true;
        for (int j = std::max(0, i - window_size); j <= std::min(n - 1, i + window_size); ++j)
        {
            if (distances[j] > distances[i])
            {
                is_max = false;
                break;
            }
        }
        if (is_max)
            local_maxima.push_back(i);
    }
    return local_maxima;
}

//-------------------------

std::vector<std::pair<int, int>> pair_indices_symmetric(const std::vector<int>& indices)
{
    std::vector<std::pair<int, int>> pairs;
    if (indices.size() % 2 != 0)
    {
        std::cerr << "Errore: numero dispari di massimi locali (" << indices.size() << ")" << std::endl;
        return pairs;
    }

    int n = indices.size();
    for (int i = 0; i < n / 2; ++i)
    {
        pairs.emplace_back(indices[i], indices[n - 1 - i]);
    }

    return pairs;
}


std::vector<std::vector<cv::Point>> extract_points_from_pairs(const std::vector<std::vector<cv::Point>>& cards, const std::vector<std::vector<cv::Point>>& ext_pts)
{
	std::vector<std::vector<cv::Point>> corner_pts;
	cv::Point bl, br, tr, tl;

	std::vector<std::pair<int, int>> paired_indices;
	std::vector<int> max_indices;
	int i = 0, j = 0;

	const int PIXEL_TOLERANCE = 3;

	std::vector<double> distances;
	double line_length;

	for (auto &card : cards)
	{
		bool first = true;

		bl = cv::Point(ext_pts[i][0].x - PIXEL_TOLERANCE, ext_pts[i][0].y + PIXEL_TOLERANCE);
		tr = cv::Point(ext_pts[i][1].x + PIXEL_TOLERANCE, ext_pts[i][1].y - PIXEL_TOLERANCE);

		i++;

		line_length = cv::norm(tr - bl);
		
		distances.clear();
		distances.reserve(card.size());

		for (const auto& pt : card)
			distances.push_back(point_line_distance(pt, bl, tr, line_length));

		max_indices = find_local_maxima(distances, 20);
		
		paired_indices = pair_indices_symmetric(max_indices);
		
		std::cout << "Card " << i << ": " << paired_indices.size() << " coppie di estremi trovate." << std::endl;

		if (paired_indices.size() == 1)
		{
			br = card[paired_indices[0].first];
			tl = card[paired_indices[0].second];
			corner_pts.push_back({ bl, br, tr, tl });
		}
		else if (paired_indices.size() > 1)
		{
			for (j = 0; j < paired_indices.size() - 1; ++j)
			{
				br = cv::Point(card[paired_indices[j].first].x + PIXEL_TOLERANCE, card[paired_indices[j].first].y + PIXEL_TOLERANCE);
				tl = cv::Point(card[paired_indices[j].second].x - PIXEL_TOLERANCE, card[paired_indices[j].second].y - PIXEL_TOLERANCE);

				if (first)
				{
					first = false;
					corner_pts.push_back({ bl, br, cv::Point(br.x, tl.y), tl });
				}
				else
				{
					corner_pts.push_back({ cv::Point(tl.x, br.y), br, cv::Point(br.x, tl.y), tl });
				}
			}

			br = cv::Point(card[paired_indices[j].first].x + PIXEL_TOLERANCE, card[paired_indices[j].first].y + PIXEL_TOLERANCE);
			tl = cv::Point(card[paired_indices[j].second].x - PIXEL_TOLERANCE, card[paired_indices[j].second].y - PIXEL_TOLERANCE);

			// Ultimo blocco: usa l'ultimo tl e br
			corner_pts.push_back({ cv::Point(tl.x, br.y), br, tr, tl });
		}
	}

	return corner_pts;
}


std::vector<std::vector<cv::Point>> process(cv::Mat &image)
{

	std::vector<std::vector<cv::Point>> contours, cards, ext_points, lines;

	if (image.empty())
	{
		std::cout << "Image is empty" << std::endl;
		return {};
	}

	cv::findContours(image.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	cards = filter_contours(contours, 3000, 250);

	for (auto &card : cards)
	{
		auto corners = find_closest_to_corners(card, image.size());
		ext_points.push_back(corners);

		reorder_contour_with_bottom_left_first(card, corners[0]);
	}
	
	return extract_points_from_pairs(cards, ext_points);

}


void sharpen_image(cv::Mat &image)
{
	cv::Mat sharpen = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	cv::filter2D(image, image, image.depth(), sharpen);
}

void sort_corners(std::vector<cv::Point2f> &pts)
{
	// Sum and diff methods
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

	// 1) Convert to Point2f
	std::vector<cv::Point2f> srcPts;
	srcPts.reserve(4);
	for (const auto &p : quad)
		srcPts.emplace_back(float(p.x), float(p.y));

	// 2) Order them: tl, tr, br, bl
	sort_corners(srcPts);

	// 3) Define destination corners in the same order
	std::vector<cv::Point2f> dstPts = {
		cv::Point2f(0.0f, 0.0f),
		cv::Point2f(float(dstSize.width - 1), 0.0f),
		cv::Point2f(float(dstSize.width - 1), float(dstSize.height - 1)),
		cv::Point2f(0.0f, float(dstSize.height - 1))};

	cv::Mat H = cv::getPerspectiveTransform(srcPts, dstPts);

	cv::Mat warped;
	warpPerspective(src, warped, H, dstSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

	return warped;
}

std::vector<cv::Mat> get_cards(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &rects)
{
	std::vector<cv::Mat> cards;

	cv::Size card_size(400, 600);
	for (auto &rect : rects)
	{
		cv::Mat card = warp_to_rect(src, rect, card_size);
		cv::Mat rot_matrix = cv::getRotationMatrix2D(cv::Point2f(card.cols / 2, card.rows / 2), 180, 1);
		cv::warpAffine(card, card, rot_matrix, cv::Size(card.cols, card.rows));

		preprocess_card(card);
		
		cards.push_back(card);
	}

	return cards;
}

