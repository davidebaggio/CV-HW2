#include "preprocess.hpp"

void preprocessing_card(cv::Mat &image)
{
    if (image.empty())
    {
        std::cout << "Image is empty" << std::endl;
        return;
    }

		cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

		// 3. CLAHE per migliorare il contrasto
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
		clahe->apply(image, image);

		// 4. Sharpening
		cv::Mat kernel = (cv::Mat_<float>(3,3) << 
			0, -1, 0,
			-1,  5, -1,
			0, -1, 0);
		cv::filter2D(image, image, -1, kernel);

		// 5. Riduzione del rumore
		cv::GaussianBlur(image, image, cv::Size(3, 3), 0);

		// 6. Binarizzazione (ottima per OCR)
		cv::threshold(image, image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		//cv::imshow("warped", card);
		//cv::waitKey(0);
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
