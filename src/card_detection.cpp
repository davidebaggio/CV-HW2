#include "card_detection.hpp"

Mat flattener(Mat image, vector<Point2f> pts, int w, int h)
{
	Point2f temp_rect[4];
	float s[4];
	for (int i = 0; i < 4; i++)
		s[i] = pts[i].x + pts[i].y;
	Point2f tl = pts[min_element(s, s + 4) - s];
	Point2f br = pts[max_element(s, s + 4) - s];

	float d[4];
	for (int i = 0; i < 4; i++)
		d[i] = pts[i].y - pts[i].x;
	Point2f tr = pts[min_element(d, d + 4) - d];
	Point2f bl = pts[max_element(d, d + 4) - d];

	if (w <= 0.8 * h)
	{
		temp_rect[0] = tl;
		temp_rect[1] = tr;
		temp_rect[2] = br;
		temp_rect[3] = bl;
	}
	else if (w >= 1.2 * h)
	{
		temp_rect[0] = bl;
		temp_rect[1] = tl;
		temp_rect[2] = tr;
		temp_rect[3] = br;
	}
	else
	{
		temp_rect[0] = pts[1];
		temp_rect[1] = pts[0];
		temp_rect[2] = pts[3];
		temp_rect[3] = pts[2];
	}

	Point2f dst[4] = {{0, 0}, {199, 0}, {199, 299}, {0, 299}};
	Mat M = getPerspectiveTransform(temp_rect, dst);
	Mat warp;
	warpPerspective(image, warp, M, Size(200, 300));
	cvtColor(warp, warp, COLOR_BGR2GRAY);
	return warp;
}

Mat preprocess_image(Mat image)
{
	Mat gray, blur, thresh;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(5, 5), 0);
	int bkg_level = gray.at<uchar>(image.rows / 100, image.cols / 2);
	int thresh_level = bkg_level + BKG_THRESH;
	threshold(blur, thresh, thresh_level, 255, THRESH_BINARY);
	return thresh;
}

void find_cards(Mat thresh_image, vector<vector<Point>> &cnts_sort, vector<int> &cnt_is_card)
{
	vector<vector<Point>> cnts;
	vector<Vec4i> hier;
	findContours(thresh_image, cnts, hier, RETR_TREE, CHAIN_APPROX_SIMPLE);

	vector<int> index_sort(cnts.size());
	iota(index_sort.begin(), index_sort.end(), 0);
	sort(index_sort.begin(), index_sort.end(), [&cnts](int a, int b)
		 { return contourArea(cnts[a]) > contourArea(cnts[b]); });

	for (int i : index_sort)
	{
		cnts_sort.push_back(cnts[i]);
		cnt_is_card.push_back(0);
	}

	for (size_t i = 0; i < cnts_sort.size(); ++i)
	{
		double size = contourArea(cnts_sort[i]);
		double peri = arcLength(cnts_sort[i], true);
		vector<Point> approx;
		approxPolyDP(cnts_sort[i], approx, 0.01 * peri, true);
		if (size < CARD_MAX_AREA && size > CARD_MIN_AREA && hier[i][3] == -1 && approx.size() == 4)
		{
			cnt_is_card[i] = 1;
		}
	}
}

Query_card preprocess_card(vector<Point> contour, Mat image)
{
	Query_card qCard;
	qCard.contour = contour;

	double peri = arcLength(contour, true);
	vector<Point> approx;
	approxPolyDP(contour, approx, 0.01 * peri, true);
	vector<Point2f> pts;
	for (auto &pt : approx)
		pts.push_back(Point2f(pt));
	qCard.corner_pts = pts;

	Rect bound = boundingRect(contour);
	qCard.width = bound.width;
	qCard.height = bound.height;

	Point2f average(0, 0);
	for (auto &pt : pts)
		average += pt;
	average *= (1.0 / pts.size());
	qCard.center = Point((int)average.x, (int)average.y);

	qCard.warp = flattener(image, pts, qCard.width, qCard.height);
	Mat Qcorner = qCard.warp(Rect(0, 0, CORNER_WIDTH, CORNER_HEIGHT));
	Mat Qcorner_zoom;
	resize(Qcorner, Qcorner_zoom, Size(), 4, 4);

	int white_level = Qcorner_zoom.at<uchar>(15, (CORNER_WIDTH * 4) / 2);
	int thresh_level = max(1, white_level - CARD_THRESH);

	Mat query_thresh;
	threshold(Qcorner_zoom, query_thresh, thresh_level, 255, THRESH_BINARY_INV);

	Mat Qrank = query_thresh(Rect(0, 20, 128, 165));
	Mat Qsuit = query_thresh(Rect(0, 186, 128, 150));

	vector<vector<Point>> Qrank_cnts;
	findContours(Qrank, Qrank_cnts, RETR_TREE, CHAIN_APPROX_SIMPLE);
	sort(Qrank_cnts.begin(), Qrank_cnts.end(), [](auto &a, auto &b)
		 { return contourArea(a) > contourArea(b); });

	if (!Qrank_cnts.empty())
	{
		Rect bbox = boundingRect(Qrank_cnts[0]);
		resize(Qrank(bbox), qCard.rank_img, Size(RANK_WIDTH, RANK_HEIGHT));
	}

	vector<vector<Point>> Qsuit_cnts;
	findContours(Qsuit, Qsuit_cnts, RETR_TREE, CHAIN_APPROX_SIMPLE);
	sort(Qsuit_cnts.begin(), Qsuit_cnts.end(), [](auto &a, auto &b)
		 { return contourArea(a) > contourArea(b); });

	if (!Qsuit_cnts.empty())
	{
		Rect bbox = boundingRect(Qsuit_cnts[0]);
		resize(Qsuit(bbox), qCard.suit_img, Size(SUIT_WIDTH, SUIT_HEIGHT));
	}

	return qCard;
}

void match_card(Query_card &qCard, vector<Train_ranks> &train_ranks, vector<Train_suits> &train_suits)
{
	int best_rank_match_diff = 10000;
	int best_suit_match_diff = 10000;
	string best_rank_name = "Unknown";
	string best_suit_name = "Unknown";

	if (!qCard.rank_img.empty() && !qCard.suit_img.empty())
	{
		for (auto &Trank : train_ranks)
		{
			Mat diff_img;
			absdiff(qCard.rank_img, Trank.img, diff_img);
			int rank_diff = sum(diff_img)[0] / 255;
			if (rank_diff < best_rank_match_diff)
			{
				best_rank_match_diff = rank_diff;
				best_rank_name = Trank.name;
			}
		}

		for (auto &Tsuit : train_suits)
		{
			Mat diff_img;
			absdiff(qCard.suit_img, Tsuit.img, diff_img);
			int suit_diff = sum(diff_img)[0] / 255;
			if (suit_diff < best_suit_match_diff)
			{
				best_suit_match_diff = suit_diff;
				best_suit_name = Tsuit.name;
			}
		}
	}

	if (best_rank_match_diff < RANK_DIFF_MAX)
		qCard.best_rank_match = best_rank_name;
	if (best_suit_match_diff < SUIT_DIFF_MAX)
		qCard.best_suit_match = best_suit_name;

	qCard.rank_diff = best_rank_match_diff;
	qCard.suit_diff = best_suit_match_diff;
}

void draw_results(Mat &image, const Query_card &qCard)
{
	int x = qCard.center.x;
	int y = qCard.center.y;
	circle(image, Point(x, y), 5, Scalar(255, 0, 0), -1);

	putText(image, qCard.best_rank_match + " of", Point(x - 60, y - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 3);
	putText(image, qCard.best_rank_match + " of", Point(x - 60, y - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(50, 200, 200), 2);
	putText(image, qCard.best_suit_match, Point(x - 60, y + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 3);
	putText(image, qCard.best_suit_match, Point(x - 60, y + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(50, 200, 200), 2);
}
