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

	// std::unordered_map<std::string, cv::Mat> rankDesc;
	// build_catalogue_sift(rankDesc);
	// std::unordered_map<std::string, cv::Mat> rankTemp;
	// build_catalogue_tm(rankTemp);

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

	static std::vector<std::vector<cv::Point>> lastValidRects;
	static std::vector<std::string> lastValidTexts;

	while (true)
	{
		if (!cap.read(frame))
		{
			std::cout << "End of video or cannot read frame\n";
			break;
		}

		/*
		if (frameCount < 1000)
		{
			frameCount++;
			continue;
		}
		*/

		cv::Mat full_frame = frame.clone(); // salva frame completo per disegno e salvataggio

		int y = frame.rows / 2;
		int x = frame.cols / 2;
		int h = int(0.6 * y);
		int w = int(0.7 * x);

		// estrai ROI centrale
		cv::Rect roiRect(x - w, y - h, 2 * w, 2 * h);
		cv::Mat roi = frame(roiRect);

		if (frameCount % 5 == 0)
		{
			s_pp = roi.clone();
			l_pp = roi.clone();

			preprocessing_strong(s_pp);
			preprocessing_light(l_pp);
			rects = process(s_pp);

			cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8U);
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

				if (blackRatio > 0.4 || blackPixels == 0)
					continue;

				std::string txt = recognize_cards(rank_patch);
				// std::string txt = recognize_card_sift(rank_patch, rankDesc);

				// trasla i punti della ROI per adattarli al full_frame
				std::vector<cv::Point> translated;
				for (const auto& pt : rects[i])
					translated.emplace_back(pt.x + roiRect.x, pt.y + roiRect.y);

				validRects.push_back(translated);
				validTexts.push_back(txt);
			}

			lastValidRects = validRects;
			lastValidTexts = validTexts;
		}

		// disegna i risultati sulla versione completa del frame
		cv::drawContours(full_frame, lastValidRects, -1, cv::Scalar(255, 0, 0));
		for (size_t i = 0; i < lastValidRects.size(); ++i)
		{
			putText(full_frame, lastValidTexts[i], lastValidRects[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
		}

		cv::imshow("Original", full_frame);
		all_frames.push_back(full_frame);

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
