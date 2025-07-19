#include "process.hpp"

std::pair<cv::Point, cv::Point> find_closest_to_corners(const std::vector<cv::Point> &card, const cv::Size &image_size)
{
    cv::Point topRightCorner(image_size.width - 1, 0);
    cv::Point bottomLeftCorner(0, image_size.height - 1);

    cv::Point closestToTopRight(-1, -1);
    cv::Point closestToBottomLeft(-1, -1);
    double minDistTopRight = std::numeric_limits<double>::max();
    double minDistBottomLeft = std::numeric_limits<double>::max();

    
	for (const auto &pt : card)
	{
		double distTopRight = cv::norm(pt - topRightCorner);
		if (distTopRight < minDistTopRight)
		{
			minDistTopRight = distTopRight;
			closestToTopRight = pt;
		}

		double distBottomLeft = cv::norm(pt - bottomLeftCorner);
		if (distBottomLeft < minDistBottomLeft)
		{
			minDistBottomLeft = distBottomLeft;
			closestToBottomLeft = pt;
		}
	}
    

    return {closestToBottomLeft, closestToTopRight};
}

std::pair<cv::Point, cv::Point> get_rotated_extremes(const std::vector<cv::Point>& contour)
{
    if (contour.empty())
        return {cv::Point(-1, -1), cv::Point(-1, -1)};

    cv::RotatedRect box = cv::minAreaRect(contour);
    cv::Point2f corners[4];
    box.points(corners);

    cv::Point topRight = corners[0];
    cv::Point bottomLeft = corners[0];

    for (int i = 1; i < 4; ++i)
    {
        const cv::Point& pt = corners[i];

        // Top-right: più alto (y min), in caso di pareggio più a destra (x max)
        if ((pt.y < topRight.y) || (pt.y == topRight.y && pt.x > topRight.x))
            topRight = pt;

        // Bottom-left: più basso (y max), in caso di pareggio più a sinistra (x min)
        if ((pt.y > bottomLeft.y) || (pt.y == bottomLeft.y && pt.x < bottomLeft.x))
            bottomLeft = pt;
    }

    return {topRight, bottomLeft};
}

std::vector<std::pair<cv::Point, cv::Point>> get_all_rotated_extreme_points(const std::vector<std::vector<cv::Point>>& contours)
{
    std::vector<std::pair<cv::Point, cv::Point>> extremes;

    for (const auto& contour : contours)
    {
        extremes.push_back(get_rotated_extremes(contour));
    }

    return extremes;
}


// Funzione per ottenere una linea tra due punti (Bresenham)
std::vector<cv::Point> get_line_points(const cv::Point& p1, const cv::Point& p2, cv::Mat &image)
{
    std::vector<cv::Point> linePoints;
    cv::LineIterator it(image, p1, p2, 8);

    for (int i = 0; i < it.count; ++i, ++it)
    {
        linePoints.push_back(it.pos());
    }
    return linePoints;
}

// Funzione che accetta più coppie di estremi
std::vector<std::vector<cv::Point>> get_lines_from_extremes(const std::vector<std::pair<cv::Point, cv::Point>>& extremes, cv::Mat &image)
{
    std::vector<std::vector<cv::Point>> allLines;

    for (const auto& pair : extremes)
    {
        const cv::Point& p1 = pair.first;
        const cv::Point& p2 = pair.second;

        if (p1 == cv::Point(-1, -1) || p2 == cv::Point(-1, -1))
        {
            allLines.push_back({});  // linea vuota per contorno vuoto
        }
        else
        {
            allLines.push_back(get_line_points(p1, p2, image));
        }
    }

    return allLines;
}


std::vector<std::vector<cv::Point>> process(cv::Mat &image)
{
	if (image.empty())
	{
		std::cout << "Image is empty" << std::endl;
		return std::vector<std::vector<cv::Point>>();
	}
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	std::vector<std::vector<cv::Point>> polys, cards;
	for (auto &contour : contours)
	{
		std::vector<cv::Point> hull;
		cv::convexHull(contour, hull);
		double area = cv::contourArea(hull);
		double peri = cv::arcLength(hull, true);
		if (area < 3000 || peri < 250)
			continue;

		
		// std::cout << "Area: " << area << std::endl;
		// std::cout << "Peri: " << peri << std::endl;
		std::vector<cv::Point> approx;
		cv::approxPolyDP(hull, approx, 0.012 * peri, true);
		std::vector<cv::Point> quad;
		if (approx.size() == 4)
		{
			quad.assign(approx.begin(), approx.end());
			polys.push_back(approx);
			cards.push_back(contour);
		}
		else
		{
			cv::RotatedRect mr = cv::minAreaRect(hull);
			cv::Point2f pts[4];
			mr.points(pts);
			quad = {pts, pts + 4};
			polys.push_back(quad);
			cards.push_back(contour);
		}
	}

	std::vector<std::pair<cv::Point, cv::Point>> etr_points;
	//etr_points = get_all_rotated_extreme_points(cards);
	for (const auto &card : cards)
    {
		etr_points.push_back(find_closest_to_corners(card, image.size()));
	}	


	std::vector<std::vector<cv::Point>> lines = get_lines_from_extremes(etr_points, image);


	cv::Mat drawing = cv::Mat::zeros(image.size(), CV_8UC3);


	// Disegna i contorni
	for (size_t i = 0; i < cards.size(); ++i)
	{
		cv::drawContours(drawing, cards, static_cast<int>(i), cv::Scalar(255, 255, 255), 1);
	}

	for (size_t i = 0; i < etr_points.size(); ++i)
	{
    	const auto& pair = etr_points[i];
    	std::cout << "Coppia " << i << ": "
              << "Top-Right = (" << pair.first.x << ", " << pair.first.y << "), "
              << "Bottom-Left = (" << pair.second.x << ", " << pair.second.y << ")"
              << std::endl;
	}

	// Disegna le linee
	for (const auto& line : lines)
	{
		for (const auto& pt : line)
		{
			if (pt.y >= 0 && pt.y < drawing.rows && pt.x >= 0 && pt.x < drawing.cols)
				drawing.at<cv::Vec3b>(pt) = cv::Vec3b(255, 0, 0);  // Blu
		}
	}
	

	// Mostra risultato
	cv::imshow("Contorni + Linee", drawing);
	cv::waitKey(0);

	return polys;
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
		/* cv::imshow("warped", card);
		cv::waitKey(0); */
		// cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
		// cv::dilate(card, card, kernel);
		cards.push_back(card);
	}
	return cards;
}