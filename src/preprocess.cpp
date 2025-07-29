#include "preprocess.hpp"


void preprocessing_card(cv::Mat &image)
{
    if (image.empty())
    {
        std::cout << "Image is empty" << std::endl;
        return;
    }

    // Convert image to grayscale
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    // to enhance local contrast
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(image, image);

    // Apply sharpening kernel to enhance edges
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    cv::filter2D(image, image, -1, kernel);

    // Reduce noise using Gaussian blur
    cv::GaussianBlur(image, image, cv::Size(9, 9), 0);

    // Apply Otsu's thresholding to binarize the image
    cv::threshold(image, image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

}


void preprocessing_image(cv::Mat &image)
{
    if (image.empty())
    {
        std::cout << "Image is empty" << std::endl;
        return;
    }

    // Convert to HSV color space
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Create a binary mask for light, desaturated areas (white-like)
    cv::Mat mask_white;
    cv::inRange(hsv,
        cv::Scalar(0, 0, 245),     
        cv::Scalar(180, 40, 255),  
        mask_white);

    // Remove everything outside the white-like mask
    cv::Mat white_like = cv::Mat::zeros(image.size(), image.type());
    image.setTo(cv::Scalar(0, 0, 0), ~mask_white);    

    // Convert the remaining result to grayscale
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // Dilate the white regions to close small gaps
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(image, image, kernel); 

    // Find external contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // Fill all detected contours (solid white shapes)
    cv::fillPoly(image, contours, cv::Scalar(255));

    // Erode the result to smooth and shrink the shapes slightly
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::erode(image, image, kernel); 
}
