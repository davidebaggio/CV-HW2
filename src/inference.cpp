#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

const float CONFIDENCE_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.5;
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;

vector<string> class_names = {"10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S", "4C", "4D", "4H", "4S", "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S", "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S", "9C", "9D", "9H", "9S", "AC", "AD", "AH", "AS", "JC", "JD", "JH", "JS", "KC", "KD", "KH", "KS", "QC", "QD", "QH", "QS"};

/* vector<string> class_names = {
	"AS", "AC", "AD", "AH",
	"2S", "2C", "2D", "2H", "3S", "3C", "3D", "3H", "4S", "4C", "4D", "4H", "5S", "5C", "5D", "5H", "6S", "6C", "6D", "6H",
	"7S", "7C", "7D", "7H", "8S", "8C", "8D", "8H", "9S", "9C", "9D", "9H",
	"10S", "10C", "10D", "10H", "JS", "JC", "JD", "JH", "QS", "QC", "QD", "QH", "KS", "KC", "KD", "KH", "JOKER"};
 */
int main(int argc, char const *argv[])
{
	// Net net = readNetFromONNX("best.onnx");
	Net net = readNetFromTorch("best.pt");
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Mat image = imread("./data/example_default.png");
	Mat image = imread("./data/Images/Images/2C0.jpg");
	// resize(image, image, Size(INPUT_WIDTH, INPUT_HEIGHT));
	Mat blob = blobFromImage(image, 1.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
	net.setInput(blob);

	vector<Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	Mat detections = outputs[0];
	for (int i = 0; i < detections.rows; ++i)
	{
		float conf = detections.at<float>(i, 4);
		if (conf < CONFIDENCE_THRESHOLD)
			continue;

		float *data = detections.ptr<float>(i);
		int classId = static_cast<int>(data[5]);

		float cx = data[0] * image.cols;
		float cy = data[1] * image.rows;
		float w = data[2] * image.cols;
		float h = data[3] * image.rows;
		int left = static_cast<int>(cx - w / 2);
		int top = static_cast<int>(cy - h / 2);

		classIds.push_back(classId);
		confidences.push_back(conf);
		boxes.emplace_back(left, top, (int)w, (int)h);
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

	for (int idx : indices)
	{
		Rect box = boxes[idx];
		rectangle(image, box, Scalar(0, 255, 0), 2);
		putText(image, class_names[classIds[idx]], Point(box.x, box.y - 5),
				FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
	}
	imshow("Detections", image);
	waitKey(0);

	return 0;
}