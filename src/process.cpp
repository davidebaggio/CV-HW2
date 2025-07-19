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

void reorder_contour_with_bottom_left_first(std::vector<cv::Point> &contour, const cv::Point &bottom_left)
{
    auto it = std::find(contour.begin(), contour.end(), bottom_left);
    if (it != contour.end())
    {
        std::rotate(contour.begin(), it, contour.end());
    }
    else
    {
        std::cerr << "Errore: punto bottom_left non trovato nel contorno." << std::endl;
    }
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
//------------------------------

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
			//quad.assign(approx.begin(), approx.end());
			//polys.push_back(approx);
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

	for (auto &card : cards)
	{
		auto corners = find_closest_to_corners(card, image.size());
		etr_points.push_back(corners);

		reorder_contour_with_bottom_left_first(card, corners.first);
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

	cv::Mat debug_image = cv::Mat::zeros(image.size(), CV_8UC3);

	for (const auto &card : cards)
	{
		int total_points = static_cast<int>(card.size());
		for (int i = 0; i < total_points; ++i)
		{
			const cv::Point &pt = card[i];

			// Gradiente lineare di rosso (da scuro a molto acceso)
			int red_intensity = static_cast<int>(255.0 * i / (total_points - 1));
			red_intensity = std::clamp(red_intensity, 0, 255);  // sicurezza

			// Disegna il punto sulla matrice (rosso con gradiente)
			if (pt.y >= 0 && pt.y < debug_image.rows && pt.x >= 0 && pt.x < debug_image.cols)
				debug_image.at<cv::Vec3b>(pt) = cv::Vec3b(0, 0, red_intensity);
		}
}
std::vector<std::pair<int, int>> paired_indices;
std::vector<std::vector<cv::Point>> temp_cards;
const int PIXEL_TOLERANCE = 5;

for (int idx = 0; idx < cards.size(); ++idx)
{
    auto& contour = cards[idx];
    auto [p1, p2] = etr_points[idx];
    double line_length = cv::norm(p2 - p1);

    std::vector<double> distances;
    distances.reserve(contour.size());

    for (const auto& pt : contour)
        distances.push_back(point_line_distance(pt, p1, p2, line_length));

	std::vector<int> max_indices = find_local_maxima(distances, 10);

	paired_indices = pair_indices_symmetric(max_indices);

	// Colori distinti per ogni coppia
	std::vector<cv::Scalar> colors = {
		cv::Scalar(0, 255, 0),      // Verde
		cv::Scalar(0, 0, 255),      // Rosso
		cv::Scalar(255, 0, 0),      // Blu
		cv::Scalar(0, 255, 255),    // Giallo
		cv::Scalar(255, 0, 255),    // Magenta
		cv::Scalar(255, 255, 0),    // Ciano
		cv::Scalar(128, 128, 0),    // Oliva
		cv::Scalar(0, 128, 128),    // Teal
		cv::Scalar(128, 0, 128)     // Viola
		// Aggiungi altri colori se necessario
	};

	for (size_t k = 0; k < paired_indices.size(); ++k)
	{
		auto [idx_a, idx_b] = paired_indices[k];
		cv::Scalar color = colors[k % colors.size()];

		const cv::Point& pt_a = contour[idx_a];
		const cv::Point& pt_b = contour[idx_b];

		if (pt_a.y >= 0 && pt_a.y < drawing.rows && pt_a.x >= 0 && pt_a.x < drawing.cols)
			cv::circle(drawing, pt_a, 4, color, -1);

		if (pt_b.y >= 0 && pt_b.y < drawing.rows && pt_b.x >= 0 && pt_b.x < drawing.cols)
			cv::circle(drawing, pt_b, 4, color, -1);
	}

	temp_cards.reserve(paired_indices.size());  // Prealloca lo spazio

	for (size_t i = 0; i < paired_indices.size(); ++i)
	{
		int idx_a = paired_indices[i].first;
		int idx_b = paired_indices[i].second;

		// Accesso ai punti nel contorno
		cv::Point& pt_a = contour[idx_a];
		cv::Point& pt_b = contour[idx_b];

		if (!(pt_a.x + PIXEL_TOLERANCE > image.cols || pt_b.x - PIXEL_TOLERANCE < 0 ||
		   pt_a.y + PIXEL_TOLERANCE > image.rows || pt_b.y - PIXEL_TOLERANCE < 0))
		{
			pt_a.x = pt_a.x + PIXEL_TOLERANCE;
			pt_a.y = pt_a.y + PIXEL_TOLERANCE;
			pt_b.y = pt_b.y - PIXEL_TOLERANCE;
			pt_b.x = pt_b.x - PIXEL_TOLERANCE;
		
		}
		

		// Costruisci i punti top_right e bottom_left "incrociando" coordinate
		cv::Point top_right(pt_b.x, pt_a.y);
		cv::Point bottom_left(pt_a.x, pt_b.y);

		// Crea un quadrilatero con questi 4 punti
		std::vector<cv::Point> quad = { pt_b, top_right, pt_a, bottom_left };

		temp_cards.push_back(quad);
	}


}



cv::imshow("Ordine Punti - Gradiente Rosso", debug_image);
cv::waitKey(0);


// Mostra risultato
cv::imshow("Contorni + Linee", drawing);
cv::waitKey(0);

return temp_cards;
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
		cv::imshow("warped", card);
		cv::waitKey(0);
		// cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
		// cv::dilate(card, card, kernel);
		cards.push_back(card);
	}
	return cards;
}