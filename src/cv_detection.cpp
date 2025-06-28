#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

void set_pixel_zero(cv::Mat &src, cv::Mat &dst, size_t col, int threshold)
{
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	if (col > 2)
	{
		return;
	}
	for (size_t i = 0; i < src.rows; i++)
	{
		for (size_t j = 0; j < src.cols; j++)
		{
			if (src.at<cv::Vec3b>(i, j)[col] > threshold && gray.at<uchar>(i, j) < threshold)
				dst.at<cv::Vec3b>(i, j) = 0;
			else
				dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i, j);
		}
	}
}

void invert_pixel(cv::Mat &src, cv::Mat &dst)
{
	dst = cv::Mat::zeros(src.size(), src.type());
	for (size_t i = 0; i < src.rows; i++)
	{
		for (size_t j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
		}
	}
}

static void sortCorners(std::vector<cv::Point2f> &pts)
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

cv::Mat warpQuadToRect(const cv::Mat &src, const std::vector<cv::Point> &quad, const cv::Size &dstSize)
{
	CV_Assert(quad.size() == 4);

	// 1) Convert to Point2f
	std::vector<cv::Point2f> srcPts;
	srcPts.reserve(4);
	for (const auto &p : quad)
		srcPts.emplace_back(float(p.x), float(p.y));

	// 2) Order them: tl, tr, br, bl
	sortCorners(srcPts);

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

void preprocessing(cv::Mat &image)
{
	if (image.empty())
	{
		std::cout << "Image is empty" << std::endl;
		return;
	}
	// remove blue and green background of table
	for (size_t i = 220; i > 40; i -= 5)
	{
		set_pixel_zero(image, image, 0, i);
		set_pixel_zero(image, image, 1, i);
		// set_pixel_zero(image, image, 2, i);
	}
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(image, image);
	cv::Mat smooth = (cv::Mat_<float>(3, 3) << 0, (float)0.5 / 4, 0, (float)0.5 / 4, (float)2 / 4, (float)0.5 / 4, 0, (float)0.5 / 4, 0);
	cv::filter2D(image, image, image.depth(), smooth);
	cv::threshold(image, image, 100, 255, cv::THRESH_BINARY);
	cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
}

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

void buildCatalogue(std::unordered_map<std::string, cv::Mat> &rankDesc, std::unordered_map<std::string, cv::Mat> &suitDesc)
{
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
	for (auto r : {"Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King"})
	{
		cv::Mat tpl = cv::imread("Card_Imgs/" + std::string(r) + ".jpg", cv::IMREAD_GRAYSCALE);
		std::vector<cv::KeyPoint> kpts;
		cv::Mat desc;
		sift->detectAndCompute(tpl, cv::noArray(), kpts, desc);
		rankDesc[r] = desc;
	}
	for (auto s : {"Hearts", "Clubs", "Diamonds", "Spades"})
	{
		cv::Mat tpl = cv::imread("Card_Imgs/" + std::string(s) + ".jpg", cv::IMREAD_GRAYSCALE);
		std::vector<cv::KeyPoint> kpts;
		cv::Mat desc;
		sift->detectAndCompute(tpl, cv::noArray(), kpts, desc);
		suitDesc[s] = desc;
	}
}

std::pair<std::string, int> detectBest(const cv::Mat &queryDesc, const std::unordered_map<std::string, cv::Mat> &catalog, cv::Ptr<cv::DescriptorMatcher> &matcher, float ratioThresh = 0.7f)
{
	std::string bestLabel;
	int bestCount = 0;
	for (auto const &entry : catalog)
	{
		const std::string &label = entry.first;
		const cv::Mat &tplDesc = entry.second;
		if (tplDesc.empty() || queryDesc.empty())
			continue;
		std::vector<std::vector<cv::DMatch>> knn;
		matcher->knnMatch(tplDesc, queryDesc, knn, 2);
		int good = 0;
		for (auto &m : knn)
		{
			if (m.size() >= 2 && m[0].distance < ratioThresh * m[1].distance)
				good++;
		}
		if (good > bestCount)
		{
			bestCount = good;
			bestLabel = label;
		}
	}
	return make_pair(bestLabel, bestCount);
}

std::pair<std::string, std::string> detectCard(const cv::Mat &queryImg, std::unordered_map<std::string, cv::Mat> &rankDesc, std::unordered_map<std::string, cv::Mat> &suitDesc, int &rankScore, int &suitScore, std::vector<cv::KeyPoint> &outQueryKpts)
{
	// Extract descriptors from query
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
	cv::Mat qDesc;
	sift->detectAndCompute(queryImg, cv::noArray(), outQueryKpts, qDesc);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();

	// Detect best rank and suit
	std::pair<std::string, int> rankRes = detectBest(qDesc, rankDesc, matcher);
	std::pair<std::string, int> suitRes = detectBest(qDesc, suitDesc, matcher);
	rankScore = rankRes.second;
	suitScore = suitRes.second;
	return make_pair(rankRes.first, suitRes.first);
}

int main(int argc, char **argv)
{
	// 1. Open a video file or camera stream
	//    If you pass 0 to VideoCapture it'll open the default camera.
	std::string inputPath = (argc > 1 ? argv[1] : "your_video.mp4");
	cv::VideoCapture cap(inputPath);
	if (!cap.isOpened())
	{
		std::cerr << "ERROR: Could not open video source: " << inputPath << std::endl;
		return 1;
	}

	std::cout << "Building catalogue of SIFT descriptors" << std::endl;
	std::unordered_map<std::string, cv::Mat> rankDesc, suitDesc;
	buildCatalogue(rankDesc, suitDesc);
	std::cout << "Catalogue of SIFT descriptors builded" << std::endl;

	// 2. Retrieve video properties (optional)
	double fps = cap.get(cv::CAP_PROP_FPS);
	int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	std::cout << "Opened " << inputPath << " (" << width << "x" << height << " @ " << fps << " FPS)\n";

	cv::Mat frame;
	int frameCount = 0;

	// 3. Loop: grab each frame, process, display (or save)
	cv::Mat pp;
	std::vector<std::vector<cv::Point>> rects;
	std::vector<cv::Mat> all_frames;
	while (true)
	{
		// Read the next frame
		if (!cap.read(frame))
		{
			std::cout << "End of video or cannot read frame\n";
			break;
		}
		if (frameCount < 4000)
		{
			frameCount++;
			continue;
		}
		int y = frame.rows / 2;
		int x = frame.cols / 2;
		int h = int(0.55 * y);
		int w = int(0.7 * x);
		frame = frame(cv::Range(y - h, y + h), cv::Range(x - w, x + w));

		if (frameCount % 2 == 0)
		{
			auto begin = std::chrono::high_resolution_clock::now();
			pp = frame.clone();
			preprocessing(pp);
			rects = process(pp);
			auto end = std::chrono::high_resolution_clock::now();
			auto dur = end - begin;
			auto s = (float)std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() / 1000;
			// std::cout << "Processed " << (float)1 / s << "fps" << std::endl;
		}
		frameCount++;
		cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
		fillPoly(mask, rects, cv::Scalar(255));
		cv::drawContours(frame, rects, -1, cv::Scalar(255, 0, 0));
		//  cv::Mat sharpen = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
		//  cv::filter2D(result, result, result.depth(), sharpen);

		cv::Mat result;
		// frame.copyTo(result, mask);
		pp.copyTo(result, mask);
		cv::imshow("Masked", result);
		// cv::imshow("Processed", pp);

		cv::Size card_size(300, 450);
		for (auto &rect : rects)
		{
			cv::Mat card = warpQuadToRect(result, rect, card_size);
			cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
			cv::dilate(card, card, kernel);

			int rankScore = 0, suitScore = 0;
			std::vector<cv::KeyPoint> queryKeypoints;
			auto result = detectCard(card, rankDesc, suitDesc, rankScore, suitScore, queryKeypoints);
			std::string bestRank = result.first;
			std::string bestSuit = result.second;
			// drawKeypoints(card, queryKeypoints, card, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			/* if (rankScore >= 10 && suitScore >= 10)
			{
				std::cout << "Detected card: " << bestRank << " of " << bestSuit
						  << " (rank matches=" << rankScore
						  << ", suit matches=" << suitScore << ")" << std::endl;
			}
			else
			{
				std::cout << "Unable to confidently detect card ("
						  << "bestRank=" << bestRank << "(" << rankScore << "), "
						  << "bestSuit=" << bestSuit << "(" << suitScore << "))" << std::endl;
			} */
			std::string cardText = bestRank + " of " + bestSuit;
			cv::putText(frame, cardText, rect[0], cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1.5);

			/* cv::imshow("Card", card);
			cv::waitKey(0); */
		}

		cv::imshow("Original", frame);
		// cv::waitKey(0);
		all_frames.push_back(result);
		char key = static_cast<char>(cv::waitKey(1));
		if (key == 27 /* ESC */)
		{
			std::cout << "Interrupted by user\n";
			break;
		}
		// cv::waitKey(0);
	}

	cv::Size frameSize = all_frames[0].size(); // all frames must be same size
	bool isColor = (all_frames[0].channels() == 3);

	cv::VideoWriter writer;
	writer.open(
		"output.mp4",								 // output filename
		cv::VideoWriter::fourcc('m', 'p', '4', 'v'), // codec
		fps,										 // fps
		frameSize,									 // frame size
		isColor										 // color or grayscale
	);
	if (!writer.isOpened())
	{
		std::cerr << "Could not open the output video for write\n";
		return -1;
	}

	// 4. Write each frame
	for (const cv::Mat &frame : all_frames)
	{
		writer.write(frame);
	}
	writer.release();
	std::cout << "Saved output.mp4 (" << all_frames.size() << " frames at " << fps << " FPS)\n";
	cap.release();
	cv::destroyAllWindows();
	return 0;
}