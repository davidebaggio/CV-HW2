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
void preprocessing_light(cv::Mat &image)
{
    if (image.empty())
    {
        std::cout << "Image is empty" << std::endl;
        return;
    }

    // Rimuovi sfondo blu/verde
    for (size_t i = 220; i > 40; i -= 5)
    {
        set_pixel_zero(image, image, 0, i);
        set_pixel_zero(image, image, 1, i);
        // set_pixel_zero(image, image, 2, i);
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // Leggera sfocatura (kernel 3x3)
    cv::Mat smooth = (cv::Mat_<float>(3, 3) << 0, 0.25 / 3, 0,
                                               0.25 / 3, 2.0 / 3, 0.25 / 3,
                                               0, 0.25 / 3, 0);
    cv::filter2D(image, image, image.depth(), smooth);

    // Equalizzazione adattiva del contrasto
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(image, image);

    // Soglia adattiva (meglio in presenza di illuminazione disomogenea)
    cv::adaptiveThreshold(image, image, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY,
                          11, 8);

    // Ulteriore filtro di smoothing (opzionale)
    cv::filter2D(image, image, image.depth(), smooth);

    // Esempio di kernel morfologico (non usato qui)
    // cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0,
    //                                            1, 1, 1,
    //                                            0, 1, 0);
}

void preprocessing_strong(cv::Mat &image)
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
		cv::Scalar(0, 0, 245),     // H, S, V min: poco saturi, molto chiari
		cv::Scalar(180, 40, 255),  // H, S, V max
		mask_white);

	// Applica la maschera all’immagine
	cv::Mat white_like = cv::Mat::zeros(image.size(), image.type());
	image.setTo(cv::Scalar(0, 0, 0), ~mask_white);	
	
	// Converti in scala di grigi per uso successivo
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

	cv::Mat kernel;
	// Dilatazione
	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::dilate(image, image, kernel); 

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::fillPoly(image, contours, cv::Scalar(255));

	kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(11, 11));
	cv::erode(image, image, kernel); 

	//cv::imshow("Processed Image", image);
	//cv::waitKey(0);

}
