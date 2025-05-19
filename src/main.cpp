#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "tinyxml2.hpp"

using namespace tinyxml2;
using namespace std;

struct BoundingBox
{
	string className;
	int xmin, ymin, xmax, ymax;
};

BoundingBox parseVOC(const string &xmlPath)
{
	XMLDocument doc;
	BoundingBox box;

	if (doc.LoadFile(xmlPath.c_str()) != XML_SUCCESS)
	{
		cerr << "Failed to load " << xmlPath << endl;
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

int main()
{
	string xmlFile = "./data/Annotations/Annotations/2C0.xml";
	string imageFile = "./data/Images/Images/2C0.jpg";

	BoundingBox box = parseVOC(xmlFile);
	cv::Mat img = cv::imread(imageFile, cv::IMREAD_COLOR);

	// Draw bounding box
	cv::rectangle(img, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), cv::Scalar(0, 255, 0), 2);
	cv::putText(img, box.className, cv::Point(box.xmin, box.ymin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

	cv::namedWindow("Card", cv::WINDOW_NORMAL);
	cv::imshow("Card", img);
	cv::waitKey(0);
	return 0;
}
