// Francesco Pivotto 2158296

#include "preprocess.hpp"

void preprocessing_card(cv::Mat &image)
{
    if (image.empty())
    {
        std::cout << "Image is empty" << std::endl;
        return;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(image, image);

    // Sharpening
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0,
                      -1, 5, -1,
                      0, -1, 0);
    cv::filter2D(image, image, -1, kernel);
    cv::GaussianBlur(image, image, cv::Size(9, 9), 0);
    cv::threshold(image, image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void preprocessing_image(cv::Mat &image)
{
    if (image.empty())
    {
        std::cout << "Image is empty" << std::endl;
        return;
    }

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask_white;
    cv::inRange(hsv,
                cv::Scalar(0, 0, 245),
                cv::Scalar(180, 40, 255),
                mask_white);
    cv::Mat white_like = cv::Mat::zeros(image.size(), image.type());
    image.setTo(cv::Scalar(0, 0, 0), ~mask_white);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(image, image, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::fillPoly(image, contours, cv::Scalar(255));

    // Erode the result to smooth and shrink the shapes slightly
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::erode(image, image, kernel);
}
