#include "preprocess.hpp"

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
		//  set_pixel_zero(image, image, 2, i);
	}
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::Mat smooth = (cv::Mat_<float>(3, 3) << 0.0f, 0.25f / 3.0f, 0.0f, 0.25 / 3.0f, 2 / 3.0f, 0.25 / 3.0f, 0.0f, 0.25 / 3.0f, 0);
	// cv::filter2D(image, image, image.depth(), smooth);
	cv::equalizeHist(image, image);
	cv::threshold(image, image, 110, 255, cv::THRESH_BINARY);
	// cv::filter2D(image, image, image.depth(), smooth);
	cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
}