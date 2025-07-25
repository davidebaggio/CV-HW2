#include "detect.hpp"
#include <torch/torch.h>
#include <iostream>

// Variabili globali definite *prima* di usarle
torch::jit::script::Module card_model = load_card_model("simple_card_classifier_traced.pt");
const std::vector<std::string> card_classes = {"10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"};

// Caricamento modello
torch::jit::script::Module load_card_model(const std::string& model_path) {
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Errore nel caricamento del modello: " << e.what() << std::endl;
        throw;
    }
    return model;
}

std::string recognize_cards(const cv::Mat& rank_patch) {
    if (rank_patch.empty()) return "Invalid";

    cv::Mat resized;
    cv::resize(rank_patch, resized, cv::Size(128, 128));

    resized.convertTo(resized, CV_32F, 1.0 / 255);
    resized = (resized - 0.5) / 0.5;

    // Assumendo immagine in scala di grigi: [1,1,128,128]
    torch::Tensor input_tensor = torch::from_blob(
        resized.data, {1, 1, 128, 128}, torch::kFloat32).clone();

    torch::NoGradGuard no_grad;
    torch::Tensor output = card_model.forward({input_tensor}).toTensor();

    int pred_idx = output.argmax(1).item<int>();
    if (pred_idx < 0 || pred_idx >= static_cast<int>(card_classes.size()))
        return "Unknown";

    return card_classes[pred_idx];
}


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
		cv::Size winSize(cvRound(baseWindow.width * scale), cvRound(baseWindow.height * scale));
		if (winSize.width < 20 || winSize.height < 20)
			continue;

		cv::Rect win(0, 0, winSize.width, winSize.height);
		cv::Mat patch = gray(win);
		if (cv::countNonZero(patch) < 0.15 * patch.rows * patch.cols)
			continue;

		// Usa binarizzazione solo per trovare contorni
		cv::Mat bwBin;
		cv::threshold(patch, bwBin, 127, 255, cv::THRESH_BINARY_INV);

		// Trova contorni sulla binarizzazione
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(bwBin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		// Filtra contorni con area >= 450
		std::vector<std::vector<cv::Point>> filteredContours;
		for (const auto& contour : contours)
		{
			double area = cv::contourArea(contour);
			if (area >= 450)
				filteredContours.push_back(contour);
		}

		if (filteredContours.empty())
			continue; // Nessun contorno valido

		// Crea maschera dal primo contorno valido
		cv::Mat mask = cv::Mat::zeros(patch.size(), CV_8UC1);
		cv::drawContours(mask, filteredContours, 0, 255, cv::FILLED);

		// Ora applica la maschera su patch originale (in scala di grigi)
		cv::Mat bw(patch.size(), patch.type(), cv::Scalar(255)); // tutto bianco
		patch.copyTo(bw, mask);

		// Visualizza patch mascherato (opzionale)
		cv::imshow("Patch mascherato (bw)", bw);
		cv::waitKey(0);

		// Matching su patch mascherato
		auto [rankLabel, rankScore] = best_template_match(bw, rankTemplate, rankThresh);
		if (rankLabel.empty())
			continue;

		detections.push_back((card_corner){win, rankLabel, rankScore});
	}

	return non_max_suppression(detections, 0.1f);
}

cv::Mat extract_rank_patch(const cv::Mat& gray) {
    // Parametri fissi scelti in modo empirico
    const cv::Size baseWindow(60, 120);
    const double scale = 1.2;

	cv::Size winSize(cvRound(baseWindow.width * scale), cvRound(baseWindow.height * scale));
	
	cv::Rect win(0, 0, winSize.width, winSize.height);
    cv::Mat patch = gray(win);

    if (winSize.width < 20 || winSize.height < 20){
		cv::Mat white = cv::Mat::zeros(patch.size(), CV_8UC1);
        return white; // Nessun contorno valido, ritorna un'immagine bianca
		//throw std::runtime_error("Window size too small after scaling");
	}

    if (gray.cols < winSize.width || gray.rows < winSize.height){
		cv::Mat white = cv::Mat::zeros(patch.size(), CV_8UC1);
        return white; // Nessun contorno valido, ritorna un'immagine bianca
		//throw std::runtime_error("Input image too small for fixed window size");
		}
 

    if (cv::countNonZero(patch) < 0.15 * patch.rows * patch.cols){
		cv::Mat white = cv::Mat::zeros(patch.size(), CV_8UC1);
        return white; // Nessun contorno valido, ritorna un'immagine bianca
        //throw std::runtime_error("Patch troppo vuoto, non contiene abbastanza informazioni");
	}

    cv::Mat bwBin;
    cv::threshold(patch, bwBin, 127, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bwBin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty())
        throw std::runtime_error("Nessun contorno trovato");

    std::vector<cv::Point> selectedContour;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) >= 450) {
            selectedContour = contour;
            break;
        }
    }

    if (selectedContour.empty()){
        cv::Mat white = cv::Mat::zeros(patch.size(), CV_8UC1);
        return white; // Nessun contorno valido, ritorna un'immagine bianca
    }
    cv::Mat mask = cv::Mat::zeros(patch.size(), CV_8UC1);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{selectedContour}, 0, 255, cv::FILLED);

    cv::Mat bw(patch.size(), patch.type(), cv::Scalar(255));
    patch.copyTo(bw, mask);

    // Binarizzazione finale con soglia fissa 120
    cv::Mat final_bin;
    cv::threshold(bw, final_bin, 120, 255, cv::THRESH_BINARY);

    cv::imshow("Patch mascherato (bw)", final_bin);
    //cv::waitKey(0);

    return final_bin;
}

cv::Mat extract_rank_patch_center_based(const cv::Mat& gray) {
    
    const cv::Size baseWindow(80, 120);
    const double scale = 1.2;
    cv::Size winSize(cvRound(baseWindow.width * scale), cvRound(baseWindow.height * scale));

    if (gray.cols < winSize.width || gray.rows < winSize.height) {
        return cv::Mat::zeros(winSize, CV_8UC1);
    }

    cv::Rect win(0, 0, winSize.width, winSize.height);
    cv::Mat patch = gray(win).clone();  // clone to avoid modifying input

    // === CONTRASTO (CLAHE) ===
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(8.0, cv::Size(8, 8));
    clahe->apply(patch, patch);

    // === BINARIZZAZIONE ===
    cv::Mat bin;
    cv::threshold(patch, bin, 180, 255, cv::THRESH_BINARY_INV);

    // === MORFOLOGIA (closing subito dopo binarizzazione) ===
    //cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) {
        return cv::Mat::zeros(winSize, CV_8UC1);
    }

    cv::Point2f center(winSize.width / 2.0f, winSize.height / 2.0f);
    double centralBoxWidth = winSize.width * 0.5;
    double centralBoxHeight = winSize.height * 0.5;
    cv::Rect centralRect(
        static_cast<int>(center.x - centralBoxWidth / 2.0),
        static_cast<int>(center.y - centralBoxHeight / 2.0),
        static_cast<int>(centralBoxWidth),
        static_cast<int>(centralBoxHeight)
    );

    std::vector<std::pair<double, int>> validContours;
    for (size_t i = 0; i < contours.size(); i++) {
        const auto& contour = contours[i];

        // === VERIFICA TOCCO BORDO ===
        bool touchesBorder = false;
        for (const auto& pt : contour) {
            if (pt.x <= 0 || pt.y <= 0 || pt.x >= bin.cols - 1 || pt.y >= bin.rows - 1) {
                touchesBorder = true;
                break;
            }
        }
        if (touchesBorder) continue;

        // === VERIFICA INTERSEZIONE CON AREA CENTRALE ===
        bool intersectsCentralRect = false;
        for (const auto& pt : contour) {
            if (centralRect.contains(pt)) {
                intersectsCentralRect = true;
                break;
            }
        }
        if (!intersectsCentralRect) continue;

        // === DISTANZA DAL CENTRO ===
        cv::Moments m = cv::moments(contour);
        if (m.m00 == 0) continue;
        cv::Point2f contour_center(m.m10 / m.m00, m.m01 / m.m00);
        double dist = cv::norm(contour_center - center);
        validContours.emplace_back(dist, static_cast<int>(i));
    }

    if (validContours.empty()) {
        return cv::Mat::ones(winSize, CV_8UC1) * 255;
    }

    std::sort(validContours.begin(), validContours.end());

    std::vector<std::vector<cv::Point>> selectedContours;
    for (size_t i = 0; i < std::min<size_t>(2, validContours.size()); ++i) {
        selectedContours.push_back(contours[validContours[i].second]);
    }

    cv::Mat mask = cv::Mat::zeros(patch.size(), CV_8UC1);
    cv::drawContours(mask, selectedContours, -1, 255, cv::FILLED);

    cv::Mat result(patch.size(), CV_8UC1, cv::Scalar(255));
    patch.copyTo(result, mask);

    cv::threshold(result, result, 234, 255, cv::THRESH_BINARY);

    cv::imshow("Patch centrato", result);
    // cv::waitKey(0);

    return result;
}	


