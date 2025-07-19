/*void set_red_to_white(cv::Mat& image)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Range HSV per rosso-rosato (circa rgba(226,34,67))
    cv::Scalar lower_red(160, 100, 50);
    cv::Scalar upper_red(175, 255, 255);

    cv::Mat mask_red;
    cv::inRange(hsv, lower_red, upper_red, mask_red);

    image.setTo(cv::Scalar(255, 255, 255), mask_red);
}

void remove_colors(cv::Mat& image)
{
	cv::Mat hsv;
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

	cv::Mat mask_total = cv::Mat::zeros(image.size(), CV_8U);

	// === Blu scuro ===
	cv::Mat mask_blue;
	cv::inRange(hsv,
		cv::Scalar(100, 100, 30),
		cv::Scalar(130, 255, 255),
		mask_blue);
	mask_total |= mask_blue;

	// === Azzurro chiaro ===
	cv::Mat mask_cyan;
	cv::inRange(hsv,
		cv::Scalar(85, 50, 100),
		cv::Scalar(105, 255, 255),
		mask_cyan);
	mask_total |= mask_cyan;

	// === Verde molto chiaro (tipo rgba(225,241,220)) ===
	cv::Mat mask_green_pale;
	cv::inRange(hsv,
		cv::Scalar(45, 10, 200),   // H ≈ 45, bassa saturazione, V alta
		cv::Scalar(75, 60, 255),
		mask_green_pale);

	mask_total |= mask_green_pale;

	// === Giallo intenso ===
	cv::Mat mask_yellow;
	cv::inRange(hsv,
		cv::Scalar(20, 100, 100),
		cv::Scalar(35, 255, 255),
		mask_yellow);
	mask_total |= mask_yellow;

	// === Giallo specifico (251,245,149) ma non beige (252,248,238) ===
	cv::Mat mask_yellow_strict;
	cv::inRange(hsv,
		cv::Scalar(25, 100, 180),   // più stretto sul valore
		cv::Scalar(30, 255, 255),
		mask_yellow_strict);
	mask_total |= mask_yellow_strict;

	// === Beige chiaro: (233,223,185) ma non bianco ===
	cv::Mat mask_beige;
	cv::inRange(hsv,
		cv::Scalar(20, 30, 160),   // H, S, V lower bound
		cv::Scalar(40, 80, 240),   // H, S, V upper bound
		mask_beige);
	mask_total |= mask_beige;

	// === Pelle / rosa chiaro ===
	cv::Mat mask_skin;
	cv::inRange(hsv,
		cv::Scalar(0, 30, 80),
		cv::Scalar(20, 100, 255),
		mask_skin);
	mask_total |= mask_skin;

	// Applica la maschera: imposta a nero i pixel che vuoi rimuovere
	image.setTo(cv::Scalar(0, 0, 0), mask_total);

	cv::imshow("Masked Image", image);
	cv::waitKey(0);
}
*/

/*
	cv::Mat total_mask = cv::Mat::zeros(image.size(), CV_8U);

	// === Giallo vivo ===
	cv::Mat mask_yellow;
	cv::inRange(hsv,
		cv::Scalar(20, 100, 100),
		cv::Scalar(35, 255, 255),
		mask_yellow);
	total_mask |= mask_yellow;

	// === Beige chiaro: rgba(252,248,238) ===
	cv::Mat mask_lyellow;
	cv::inRange(hsv,
		cv::Scalar(20, 30, 170),
		cv::Scalar(40, 80, 255),
		mask_lyellow);
	total_mask |= mask_lyellow;

	// === Celeste chiaro: rgba(223,237,241) ===
	cv::Mat mask_lgrey;
	cv::inRange(hsv,
		cv::Scalar(85, 10, 180),
		cv::Scalar(100, 40, 255),
		mask_lgrey);
	total_mask |= mask_lgrey;

	// === Grigio medio: rgba(136,137,137) ===
	cv::Mat mask_grey;
	cv::inRange(hsv,
		cv::Scalar(0, 0, 100),
		cv::Scalar(180, 40, 150),
		mask_grey);
	total_mask |= mask_grey;

	// === Verde oliva chiaro: rgba(149,151,98) ===
	cv::Mat mask_brown;
	cv::inRange(hsv,
		cv::Scalar(30, 30, 80),
		cv::Scalar(90, 100, 180),
		mask_brown);
	total_mask |= mask_brown;

	// === Marrone scuro / Grigio scuro: rgba(63,61,59) ===
	cv::Mat mask_dgrey;
	cv::inRange(hsv,
		cv::Scalar(0, 0, 20),
		cv::Scalar(180, 50, 80),
		mask_dgrey);
	total_mask |= mask_dgrey;

	// === rgba(80,74,48) & rgba(75,74,48) ===
	cv::Mat mask_darkbrown;
	cv::inRange(hsv,
		cv::Scalar(15, 50, 20),
		cv::Scalar(45, 100, 80),
		mask_darkbrown);
	total_mask |= mask_darkbrown;

	// === rgba(125,122,97) & rgba(148,146,97) ===
	cv::Mat mask_olive;
	cv::inRange(hsv,
		cv::Scalar(20, 30, 70),
		cv::Scalar(40, 80, 180),
		mask_olive);
	total_mask |= mask_olive;

	// === rgba(237,243,224) ===
	cv::Mat mask_palegreen;
	cv::inRange(hsv,
		cv::Scalar(30, 10, 200),
		cv::Scalar(90, 60, 255),
		mask_palegreen);
	total_mask |= mask_palegreen;

	// === Colore: rgba(200,239,244) ≈ azzurro molto chiaro ===
	cv::Mat mask_cyanlight;
	cv::inRange(hsv,
		cv::Scalar(85, 30, 200),    // HSV min
		cv::Scalar(100, 80, 255),   // HSV max
		mask_cyanlight);
	total_mask |= mask_cyanlight;

	// === Applica maschera: elimina i pixel corrispondenti ===
	image.setTo(cv::Scalar(0, 0, 0), total_mask);
*/