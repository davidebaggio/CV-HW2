#ifndef CARD_DETECTION_HPP
#define CARD_DETECTION_HPP

// Playing Card Detector Functions in C++
// Author: Evan Juras (translated to C++)
// Note: Requires OpenCV library

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

// Constants
const int BKG_THRESH = 60;
const int CARD_THRESH = 30;
const int CORNER_WIDTH = 32;
const int CORNER_HEIGHT = 84;
const int RANK_WIDTH = 70;
const int RANK_HEIGHT = 125;
const int SUIT_WIDTH = 70;
const int SUIT_HEIGHT = 100;
const int RANK_DIFF_MAX = 2000;
const int SUIT_DIFF_MAX = 700;
const int CARD_MAX_AREA = 120000;
const int CARD_MIN_AREA = 25000;

// Structures
struct Train_ranks
{
	Mat img;
	string name = "Placeholder";
};

struct Train_suits
{
	Mat img;
	string name = "Placeholder";
};

struct Query_card
{
	vector<Point> contour;
	vector<Point2f> corner_pts;
	int width = 0, height = 0;
	Point center;
	Mat warp, rank_img, suit_img;
	string best_rank_match = "Unknown";
	string best_suit_match = "Unknown";
	int rank_diff = 0;
	int suit_diff = 0;
};

Mat flattener(Mat image, vector<Point2f> pts, int w, int h);

Mat preprocess_image(Mat image);

void find_cards(Mat thresh_image, vector<vector<Point>> &cnts_sort, vector<int> &cnt_is_card);

Query_card preprocess_card(vector<Point> contour, Mat image);

void match_card(Query_card &qCard, vector<Train_ranks> &train_ranks, vector<Train_suits> &train_suits);

void draw_results(Mat &image, const Query_card &qCard);

#endif