#include <iostream>
#include <string>
#include <chrono>
#include <unordered_map>
#include "preprocess.hpp"
#include "process.hpp"
#include "detect.hpp"

int get_hi_lo_value(const std::string &card_value)
{
    static const std::unordered_map<std::string, int> count_map = {
        {"2", 1}, {"3", 1}, {"4", 1}, {"5", 1}, {"6", 1},
        {"7", 0}, {"8", 0}, {"9", 0},
        {"10", -1}, {"J", -1}, {"Q", -1}, {"K", -1}, {"A", -1}
    };

    auto it = count_map.find(card_value);
    return (it != count_map.end()) ? it->second : 0;
}

cv::Scalar get_color_for_value(int value)
{
    if (value > 0)
        return cv::Scalar(0, 180, 0);       // Green
    else if (value < 0)
        return cv::Scalar(0, 40, 255);      // Red
    else
        return cv::Scalar(220, 220, 220);   // Gray
}

int main(int argc, char **argv)
{
    std::string input_path = (argc > 1 ? argv[1] : "input_video.mp4");
    cv::VideoCapture cap(input_path);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR: Could not open video source: " << input_path << std::endl;
        return 1;
    }

    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Opened " << input_path << " (" << width << "x" << height << " @ " << fps << " FPS)\n";

    // Prepare video writer
    cv::VideoWriter writer;
    cv::Size frame_size(width, height);
    bool is_color = true;

    writer.open(
        "output.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        frame_size,
        is_color
    );

    if (!writer.isOpened())
    {
        std::cerr << "Could not open the output video for write\n";
        return -1;
    }

    cv::Mat frame;
    int frame_count = 0;
    cv::Mat preprocessed_patch;
    std::vector<std::vector<cv::Point>> rects, valid_rects;
    std::vector<std::string> valid_texts;

    static std::vector<std::vector<cv::Point>> last_valid_rects;
    static std::vector<std::string> last_valid_texts;

    // Main processing loop
    while (true)
    {
        if (!cap.read(frame))
        {
            std::cout << "End of video or cannot read frame\n";
            break;
        }

        cv::Mat full_frame = frame.clone();

        // Define Region of Interest (ROI)
        int y = frame.rows / 2;
        int x = frame.cols / 2;
        int h = int(0.6 * y);
        int w = int(0.8 * x);

        cv::Rect roi_rect(x - w, y - h, 2 * w, 2 * h);
        cv::Mat roi = frame(roi_rect);

        if (frame_count % 2 == 0)
        {
            preprocessed_patch = roi.clone();

            //Preprocess the image (e.g. grayscale, threshold)
            preprocessing_image(preprocessed_patch);

            //Detect contours or rectangles
            rects = process(preprocessed_patch);

            //Create a mask and extract detected regions
            cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8U);
            fillPoly(mask, rects, cv::Scalar(255));

            cv::Mat result;
            roi.copyTo(result, mask);

            //Optional sharpening
            sharpen_image(result);

            //Crop detected cards using contours
            std::vector<cv::Mat> cards = get_cards(result, rects);

            valid_rects.clear();
            valid_texts.clear();

            //Process each detected card
            for (size_t i = 0; i < cards.size(); i++)
            {
                // Extract central patch containing card rank
                cv::Mat rank_patch = extract_rank_patch_center_based(cards[i]);

                int total_pixels = rank_patch.rows * rank_patch.cols;
                int black_pixels = total_pixels - cv::countNonZero(rank_patch);
                double black_ratio = static_cast<double>(black_pixels) / total_pixels;

                // Filter out noise or invalid detections
                if (black_ratio > 0.4 || black_pixels < 500)
                    continue;

                //Recognize card value (e.g. OCR or classification)
                std::string text = recognize_cards(rank_patch);

                //Translate rectangle coordinates back to original image
                std::vector<cv::Point> translated;
                for (const auto &pt : rects[i])
                    translated.emplace_back(pt.x + roi_rect.x, pt.y + roi_rect.y);

                valid_rects.push_back(translated);
                valid_texts.push_back(text);
            }

            // Cache last valid detection for the next frame
            last_valid_rects = valid_rects;
            last_valid_texts = valid_texts;
        }

        // Overlay results on original frame
        for (size_t i = 0; i < last_valid_rects.size(); ++i)
        {
            int hilo_value = get_hi_lo_value(last_valid_texts[i]);
            cv::Scalar color = get_color_for_value(hilo_value);

            // Create overlay with alpha blending
            cv::Mat overlay = full_frame.clone();
            std::vector<std::vector<cv::Point>> poly{last_valid_rects[i]};
            cv::fillPoly(overlay, poly, color);

            double alpha = 0.3;
            cv::addWeighted(overlay, alpha, full_frame, 1 - alpha, 0, full_frame);

            // Draw card border
            cv::drawContours(full_frame, poly, -1, color, 1);

            // Draw card text
            cv::Rect bbox = cv::boundingRect(last_valid_rects[i]);
            cv::Point top_left = bbox.tl() + cv::Point(0, 0);
            putText(full_frame, last_valid_texts[i], top_left, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

            // Draw Hi-Lo value
            cv::Point bottom_right = bbox.br() - cv::Point(-5, 10);
            std::string hilo_str = (hilo_value > 0 ? "+" : "") + std::to_string(hilo_value);
            putText(full_frame, hilo_str, bottom_right, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        // Show result
        cv::imshow("Original", full_frame);

        // Save frame to output video
        writer.write(full_frame);

        // Exit on ESC key
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27)
        {
            std::cout << "Interrupted by user\n";
            break;
        }

        frame_count++;
    }

    // Clean up
    writer.release();
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Saved output.mp4\n";
    return 0;
}
