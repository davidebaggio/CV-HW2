#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Point2f> sortCorners(vector<Point> corners)
{
	vector<Point2f> sorted(4);
	// Sum and diff for finding top-left, bottom-right, etc.
	sort(corners.begin(), corners.end(), [](Point2f a, Point2f b)
		 { return a.y < b.y; });

	Point2f tl, tr, bl, br;
	if (corners[0].x < corners[1].x)
	{
		tl = corners[0];
		tr = corners[1];
	}
	else
	{
		tl = corners[1];
		tr = corners[0];
	}

	if (corners[2].x < corners[3].x)
	{
		bl = corners[2];
		br = corners[3];
	}
	else
	{
		bl = corners[3];
		br = corners[2];
	}

	sorted[0] = tl;
	sorted[1] = tr;
	sorted[2] = br;
	sorted[3] = bl;
	return sorted;
}

int main()
{
	// Load image
	Mat image = imread("./data/images/val/5D16.jpg");
	if (image.empty())
	{
		cout << "Could not load image!" << endl;
		return -1;
	}
	Mat original_image = image.clone();

	// Resize to width = 500 (maintaining aspect ratio)
	int new_width = 500;
	double aspect_ratio = (double)new_width / image.cols;
	int new_height = (int)(image.rows * aspect_ratio);
	resize(image, image, Size(new_width, new_height));

	// Convert to grayscale and apply Gaussian blur
	Mat gray, blur_img, thresh;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur_img, Size(3, 3), 0);

	// Otsu's thresholding
	threshold(blur_img, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);

	// Find contours
	vector<vector<Point>> contours;
	findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Filter contours by area
	int threshold_min_area = 800;
	int number_of_contours = 0;
	for (const auto &c : contours)
	{
		double area = contourArea(c);
		if (area > threshold_min_area)
		{
			drawContours(image, vector<vector<Point>>{c}, -1, Scalar(36, 255, 12), 3);
			number_of_contours++;

			vector<Point> approx;
			double eps = 0.01;
			while (approx.size() != 4)
			{
				approxPolyDP(c, approx, arcLength(c, true) * eps, true);
				eps += 0.01;
				if (approx.size() == 0)
					break;
			}

			cout << "Approx: " << approx.size() << endl;
			if (approx.size() == 4)
			{
				cout << approx[0] << " " << approx[1] << " " << approx[2] << " " << approx[3] << endl;
				vector<Point2f> ord = sortCorners(approx);
				Point2f ordered[4] = {ord[0], ord[1], ord[2], ord[3]};

				Point2f dst[4] = {Point2f(0, 0), Point2f(200, 0), Point2f(200, 300), Point2f(0, 300)};
				Mat M = getPerspectiveTransform(ordered, dst);
				Mat warped;
				warpPerspective(original_image, warped, M, Size(200, 300));
				imshow("warped", warped);
				waitKey(0);
			}
		}
	}

	cout << "Contours detected: " << number_of_contours << endl;

	imshow("thresh", thresh);
	imshow("image", image);
	waitKey(0);

	return 0;
}
