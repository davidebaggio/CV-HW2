#include "evaluation.hpp"

double compute_iou(const std::vector<cv::Point> &pred, const std::vector<cv::Point> &gt)
{
	cv::Rect r1 = cv::boundingRect(pred);
	cv::Rect r2 = cv::boundingRect(gt);
	int area_intersection = (r1 & r2).area();
	int area_union = r1.area() + r2.area() - area_intersection;
	return area_union > 0 ? static_cast<double>(area_intersection) / area_union : 0.0;
}

void evaluate_predictions(const std::string &json_path,
						  const std::map<std::string, std::vector<std::pair<std::vector<cv::Point>, std::string>>> &predictions,
						  double iou_threshold)
{
	std::ifstream file(json_path);
	nlohmann::json gt_json;
	file >> gt_json;

	std::map<int, std::string> category_id_to_name;
	for (const auto &cat : gt_json["categories"])
	{
		category_id_to_name[cat["id"]] = cat["name"];
	}

	std::map<int, std::string> image_id_to_filename;
	for (const auto &img : gt_json["images"])
	{
		image_id_to_filename[img["id"]] = img["file_name"];
	}

	std::map<std::string, std::vector<std::pair<std::vector<cv::Point>, std::string>>> ground_truths;
	for (const auto &ann : gt_json["annotations"])
	{
		int image_id = ann["image_id"];
		std::string filename = image_id_to_filename[image_id];
		std::vector<cv::Point> polygon;
		for (size_t i = 0; i < ann["segmentation"][0].size(); i += 2)
		{
			float x = ann["segmentation"][0][i];
			float y = ann["segmentation"][0][i + 1];
			polygon.emplace_back(cv::Point(x, y));
		}
		std::string label = category_id_to_name[ann["category_id"]];
		ground_truths[filename].emplace_back(polygon, label);
	}

	int correct = 0, total_pred = 0, total_gt = 0;

	for (const auto &[filename, preds] : predictions)
	{
		const auto &gts = ground_truths[filename];
		total_gt += gts.size();
		total_pred += preds.size();

		std::vector<bool> matched(gts.size(), false);

		for (const auto &[pred_poly, pred_label] : preds)
		{
			double best_iou = 0.0;
			int best_idx = -1;
			for (size_t i = 0; i < gts.size(); ++i)
			{
				if (matched[i])
					continue;
				double iou = compute_iou(pred_poly, gts[i].first);
				if (iou > best_iou)
				{
					best_iou = iou;
					best_idx = i;
				}
			}

			if (best_iou >= iou_threshold && best_idx >= 0 && pred_label == gts[best_idx].second)
			{
				matched[best_idx] = true;
				correct++;
			}
		}
	}

	double precision = (total_pred > 0) ? static_cast<double>(correct) / total_pred : 0.0;
	double recall = (total_gt > 0) ? static_cast<double>(correct) / total_gt : 0.0;
	double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;

	std::cout << "Evaluation results:\n";
	std::cout << "  Precision: " << precision << "\n";
	std::cout << "  Recall:    " << recall << "\n";
	std::cout << "  F1 Score:  " << f1 << "\n";
}

int get_hi_lo_value(const std::string &card_value)
{
	static const std::unordered_map<std::string, int> count_map = {
		{"2", 1}, {"3", 1}, {"4", 1}, {"5", 1}, {"6", 1}, {"7", 0}, {"8", 0}, {"9", 0}, {"10", -1}, {"J", -1}, {"Q", -1}, {"K", -1}, {"A", -1}};

	auto it = count_map.find(card_value);
	return (it != count_map.end()) ? it->second : 0;
}

cv::Scalar get_color_for_value(int value)
{
	if (value > 0)
		return cv::Scalar(0, 180, 0); // Green
	else if (value < 0)
		return cv::Scalar(0, 40, 255); // Red
	else
		return cv::Scalar(220, 220, 220); // Gray
}