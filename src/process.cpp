#include "process.hpp"

std::vector<std::vector<cv::Point>> process(cv::Mat &image)
{
	if (image.empty())
	{
		std::cout << "Image is empty" << std::endl;
		return std::vector<std::vector<cv::Point>>();
	}
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<std::vector<cv::Point>> polys;
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
		}
		else
		{
			cv::RotatedRect mr = cv::minAreaRect(hull);
			cv::Point2f pts[4];
			mr.points(pts);
			quad = {pts, pts + 4};
			polys.push_back(quad);
		}
	}

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
		//  cv::dilate(card, card, kernel);
		cards.push_back(card);
	}
	return cards;
}