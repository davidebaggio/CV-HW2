#include <iostream>
#include <string>
#include <chrono>
#include "preprocess.hpp"
#include "process.hpp"
#include "detect.hpp"

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
	std::unordered_map<std::string, cv::Mat> rankDesc, rankTemp;
	build_catalogue_sift(rankDesc);
	std::cout << "Catalogue of SIFT descriptors builded" << std::endl;
	std::cout << "Building catalogue of Templates" << std::endl;
	build_catalogue_tm(rankTemp);
	std::cout << "Catalogue of Templates builded" << std::endl;

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
			pp = frame.clone();
			auto begin = std::chrono::high_resolution_clock::now();
			preprocessing(pp);
			rects = process(pp);
			auto end = std::chrono::high_resolution_clock::now();
			auto dur = end - begin;
			auto s = (float)std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() / 1000;
			// std::cout << "Processed " << (float)1 / s << "fps" << std::endl;
		}
		frameCount++;
		cv::drawContours(frame, rects, -1, cv::Scalar(255, 0, 0));
		cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
		fillPoly(mask, rects, cv::Scalar(255));

		cv::Mat result;
		pp.copyTo(result, mask);
		sharpen_image(result);
		// cv::imshow("Processed", pp);

		std::vector<cv::Mat> cards = get_cards(result, rects);

		for (size_t i = 0; i < cards.size(); i++)
		{
#if 0
			int rankScore = 0, suitScore = 0;
			std::vector<cv::KeyPoint> queryKeypoints;
			auto result = detect_card_sift(cards[i], rankDesc, suitDesc, rankScore, suitScore, queryKeypoints);
			std::string bestRank = result.first;
			std::string bestSuit = result.second;
			std::string cardText = bestRank + " of " + bestSuit;
			cv::putText(frame, cardText, rects[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1.5);
#else
			std::vector<card_corner> detected = detect_with_tl_window(cards[i], rankTemp);
			if (detected.size() == 0)
			{
				detected = detect_with_sliding_window(cards[i], rankTemp);
			}
			for (auto &d : detected)
			{
				std::string txt = d.rank;
				putText(frame, txt, d.window.tl() + rects[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1.5);
			}
#endif
			// cv::imshow("Card", cards[i]);
			// cv::waitKey(0);
		}

		cv::imshow("Masked", result);
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