#include "detect.hpp"

static float IoU(const cv::Rect &a, const cv::Rect &b)
{
	int x1 = std::max(a.x, b.x);
	int y1 = std::max(a.y, b.y);
	int x2 = std::min(a.x + a.width, b.x + b.width);
	int y2 = std::min(a.y + a.height, b.y + b.height);
	int w = std::max(0, x2 - x1);
	int h = std::max(0, y2 - y1);
	float inter = w * h;
	float uni = float(a.area() + b.area() - inter);
	return uni > 0.f ? inter / uni : 0.f;
}

std::vector<card_corner> non_max_suppression(
	std::vector<card_corner> &dets,
	float iouThresh)
{
	// Compute a single confidence per detection
	struct Item
	{
		card_corner d;
		double score;
	};
	std::vector<Item> items;
	items.reserve(dets.size());
	for (auto &d : dets)
	{
		// e.g. average of both scores, or max; here sum:
		double conf = d.rankScore;
		items.push_back({d, conf});
	}

	// Sort descending by confidence
	sort(items.begin(), items.end(),
		 [](auto &a, auto &b)
		 { return a.score > b.score; });

	std::vector<card_corner> result;
	std::vector<bool> suppressed(items.size(), false);

	for (size_t i = 0; i < items.size(); ++i)
	{
		if (suppressed[i])
			continue;
		// Keep the highest-score box
		result.push_back(items[i].d);
		// Suppress any with high overlap
		for (size_t j = i + 1; j < items.size(); ++j)
		{
			if (suppressed[j])
				continue;
			if (IoU(items[i].d.window, items[j].d.window) > iouThresh)
			{
				suppressed[j] = true;
			}
		}
	}
	return result;
}

void build_catalogue_sift(std::unordered_map<std::string, cv::Mat> &rankDesc)
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
}

void build_catalogue_tm(std::unordered_map<std::string, cv::Mat> &rankTemplate)
{
	for (auto r : {"Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King"})
	{
		cv::Mat tpl = cv::imread("Card_Imgs/" + std::string(r) + ".jpg", cv::IMREAD_GRAYSCALE);
		if (tpl.empty())
			continue;
		rankTemplate[r] = tpl;
	}
}

std::pair<std::string, double> best_template_match(const cv::Mat &patch, const std::unordered_map<std::string, cv::Mat> &templates, double thresh)
{
	std::string bestLabel = "";
	double bestScore = thresh;
	for (auto &kv : templates)
	{
		const std::string &label = kv.first;
		const cv::Mat &tpl = kv.second;
		cv::Mat query = patch.clone();
		cv::resize(query, query, tpl.size(), 0, 0, cv::INTER_LINEAR);
		cv::Mat result;
		cv::matchTemplate(query, tpl, result, cv::TM_CCOEFF);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
		if (maxVal > bestScore)
		{
			bestScore = maxVal;
			bestLabel = label;
		}
	}
	return {bestLabel, bestScore};
}

std::pair<std::string, int> detect_best(const cv::Mat &queryDesc, const std::unordered_map<std::string, cv::Mat> &catalog, cv::Ptr<cv::DescriptorMatcher> &matcher, float ratioThresh)
{
	std::string bestLabel;
	int bestCount = 0;
	for (auto const &entry : catalog)
	{
		const std::string &label = entry.first;
		const cv::Mat &tplDesc = entry.second;
		if (tplDesc.empty() || queryDesc.empty() || tplDesc.rows < 2)
			continue;
		std::vector<std::vector<cv::DMatch>> knn;
		matcher->knnMatch(queryDesc, tplDesc, knn, 2);
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

std::string detect_card_sift(const cv::Mat &queryImg, std::unordered_map<std::string, cv::Mat> &rankDesc, int &rankScore, std::vector<cv::KeyPoint> &outQueryKpts)
{
	// Extract descriptors from query
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
	cv::Mat qDesc;
	sift->detectAndCompute(queryImg, cv::noArray(), outQueryKpts, qDesc);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();

	// Detect best rank and suit
	std::pair<std::string, int> rankRes = detect_best(qDesc, rankDesc, matcher);
	rankScore = rankRes.second;
	return rankRes.first;
}

std::vector<card_corner> detect_with_sliding_window(const cv::Mat &gray,
													std::unordered_map<std::string, cv::Mat> &rankTemplate,
													std::vector<double> scales,
													cv::Size baseWindow,
													int stride,
													double rankThresh)
{
	std::vector<card_corner> detections;

	for (double scale : scales)
	{
		// Compute scaled window size
		cv::Size winSize(cvRound(baseWindow.width * scale), cvRound(baseWindow.height * scale));
		if (winSize.width < 20 || winSize.height < 20)
			continue;

		for (int y = 0; y + winSize.height <= gray.rows; y += stride)
		{
			for (int x = 0; x + winSize.width <= gray.cols; x += stride)
			{
				cv::Rect win(x, y, winSize.width, winSize.height);
				cv::Mat patch = gray(win);
				// Binarize to match your templates
				cv::Mat bw;
				cv::threshold(patch, bw, 127, 255, cv::THRESH_BINARY_INV);

				// 1) rank match
				auto [rankLabel, rankScore] = best_template_match(bw, rankTemplate, rankThresh);
				if (rankLabel.empty())
					continue;

				detections.push_back((card_corner){win, rankLabel, rankScore});
			}
		}
	}

	return non_max_suppression(detections, 0.1f);
	// return detections;
}

std::vector<card_corner> detect_with_tl_window(const cv::Mat &gray,
											   std::unordered_map<std::string, cv::Mat> &rankTemplate,
											   std::vector<double> scales,
											   cv::Size baseWindow,
											   double rankThresh)
{
	std::vector<card_corner> detections;

	for (double scale : scales)
	{
		// Compute scaled window size
		cv::Size winSize(cvRound(baseWindow.width * scale), cvRound(baseWindow.height * scale));
		if (winSize.width < 20 || winSize.height < 20)
			continue;

		cv::Rect win(0, 0, winSize.width, winSize.height / 2);
		cv::Mat patch = gray(win);
		if (cv::countNonZero(patch) < 0.15 * patch.rows * patch.cols)
			continue;
		// Binarize to match your templates
		cv::Mat bw;
		cv::threshold(patch, bw, 127, 255, cv::THRESH_BINARY_INV);
		// cv::imshow("bw", bw);
		// cv::waitKey(0);

		// 1) rank match
		auto [rankLabel, rankScore] = best_template_match(bw, rankTemplate, rankThresh);
		if (rankLabel.empty())
			continue;

		detections.push_back((card_corner){win, rankLabel, rankScore});
	}

	return non_max_suppression(detections, 0.1f);
	// return detections;
}