#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "tinyxml2.hpp"

using namespace tinyxml2;

struct BoundingBox
{
	std::string className;
	int xmin, ymin, xmax, ymax;
};

BoundingBox parseVOC(const std::string &xmlPath)
{
	XMLDocument doc;
	BoundingBox box;

	if (doc.LoadFile(xmlPath.c_str()) != XML_SUCCESS)
	{
		std::cerr << "[ERROR]: Failed to load " << xmlPath << std::endl;
		return box;
	}

	XMLElement *root = doc.FirstChildElement("annotation");
	XMLElement *object = root->FirstChildElement("object");
	XMLElement *name = object->FirstChildElement("name");
	XMLElement *bndbox = object->FirstChildElement("bndbox");

	box.className = name->GetText();

	bndbox->FirstChildElement("xmin")->QueryIntText(&box.xmin);
	bndbox->FirstChildElement("ymin")->QueryIntText(&box.ymin);
	bndbox->FirstChildElement("xmax")->QueryIntText(&box.xmax);
	bndbox->FirstChildElement("ymax")->QueryIntText(&box.ymax);

	return box;
}

std::string get_filename(std::string path)
{
	std::string filename = path.substr(path.find_last_of("/") + 1);
	return filename.substr(0, filename.find_last_of("."));
}

int main()
{
	std::vector<std::string> images_paths;
	// images_paths.push_back("./data/example_default.png");
	cv::glob("./data/Images/Images/*.jpg", images_paths);

	cv::namedWindow("Card", cv::WINDOW_NORMAL);

	for (size_t i = 0; i < images_paths.size(); i++)
	{
		std::string image_file = images_paths[i];
		std::string xml_file = "./data/Annotations/Annotations/" + get_filename(image_file) + ".xml";
		BoundingBox box = parseVOC(xml_file);
		cv::Mat img = cv::imread(image_file, cv::IMREAD_COLOR);
		if (img.empty())
		{
			std::cerr << "[ERROR]: Failed to load " << image_file << std::endl;
			continue;
		}

		cv::Mat gray, blurred, thresh, edged;
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		cv::threshold(gray, thresh, 150, 255, cv::THRESH_BINARY);

		// Find contours again
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (const auto &contour : contours)
		{
			double area = cv::contourArea(contour);
			if (area < 500)
				continue; // Filter out noise

			std::vector<cv::Point> approx;
			cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);

			if (approx.size() >= 3 && approx.size() <= 4 && cv::isContourConvex(approx))
			{
				cv::polylines(img, approx, true, cv::Scalar(0, 255, 0), 5);
			}
		}

		// Draw bounding box
		cv::rectangle(img, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), cv::Scalar(0, 0, 255), 3);
		cv::putText(img, box.className, cv::Point(box.xmin, box.ymin - 30), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 0, 0), 10);

		cv::resize(img, img, cv::Size(800, 600));
		cv::resize(thresh, thresh, cv::Size(800, 600));
		cv::imshow("Card", thresh);
		cv::waitKey(0);
		cv::imshow("Card", img);
		cv::waitKey(0);
	}

	return 0;
}
