#include <iostream>
#include <string>
#include <chrono>
#include "preprocess.hpp"
#include "process.hpp"
#include "detect.hpp"

int main(int argc, char **argv)
{
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

	double fps = cap.get(cv::CAP_PROP_FPS);
	int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	std::cout << "Opened " << inputPath << " (" << width << "x" << height << " @ " << fps << " FPS)\n";

	cv::Mat frame;
	int frameCount = 0;
	cv::Mat s_pp, l_pp;
	std::vector<std::vector<cv::Point>> rects, validRects;
	std::vector<std::string> validTexts;
	std::vector<cv::Mat> all_frames;

	while (true)
	{
		if (!cap.read(frame))
		{
			std::cout << "End of video or cannot read frame\n";
			break;
		}
		if (frameCount < 1000)
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
			s_pp = frame.clone();
			l_pp = frame.clone();

			preprocessing_strong(s_pp);
			preprocessing_light(l_pp);
			rects = process(s_pp);

			cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
			fillPoly(mask, rects, cv::Scalar(255));

			cv::Mat result;
			l_pp.copyTo(result, mask);
			sharpen_image(result);

			std::vector<cv::Mat> cards = get_cards(result, rects);

			validRects.clear();
			validTexts.clear();

			for (size_t i = 0; i < cards.size(); i++)
			{
				cv::Mat rank_patch = extract_rank_patch_center_based(cards[i]);

				int totalPixels = rank_patch.rows * rank_patch.cols;
				int blackPixels = totalPixels - cv::countNonZero(rank_patch);
				double blackRatio = static_cast<double>(blackPixels) / totalPixels;

				if (blackRatio > 0.25 || blackPixels == 0)
					continue;

				std::string txt = recognize_cards(rank_patch);
				validRects.push_back(rects[i]);
				validTexts.push_back(txt);
			}
		}

		cv::drawContours(frame, validRects, -1, cv::Scalar(255, 0, 0));
		for (size_t i = 0; i < validRects.size(); ++i)
		{
			putText(frame, validTexts[i], validRects[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
		}

		cv::imshow("Original", frame);
		all_frames.push_back(frame);
		char key = static_cast<char>(cv::waitKey(1));
		if (key == 27)
		{
			std::cout << "Interrupted by user\n";
			break;
		}
		frameCount++;
	}

	cv::Size frameSize = all_frames[0].size();
	bool isColor = (all_frames[0].channels() == 3);

	cv::VideoWriter writer;
	writer.open(
		"output.mp4",
		cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
		fps,
		frameSize,
		isColor
	);
	if (!writer.isOpened())
	{
		std::cerr << "Could not open the output video for write\n";
		return -1;
	}

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
