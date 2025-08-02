// Zoren Martinez 2123873

#include "detect.hpp"

static torch::jit::script::Module card_model = load_card_model("simple_card_classifier_traced.pt");
static const std::vector<std::string> card_classes = {"10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"};

torch::jit::script::Module load_card_model(const std::string &model_path)
{
    torch::jit::script::Module model;
    try
    {
        model = torch::jit::load(model_path);
        model.eval(); // disable dropout, etc.
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Model loading error: " << e.what() << std::endl;
        throw;
    }
    return model;
}

std::string recognize_cards(const cv::Mat &rank_patch)
{
    if (rank_patch.empty())
        return "Invalid";

    cv::Mat resized;
    cv::resize(rank_patch, resized, cv::Size(128, 128));
    resized.convertTo(resized, CV_32F, 1.0 / 255);
    resized = (resized - 0.5f) / 0.5f;

    torch::Tensor input_tensor = torch::from_blob(resized.data, {1, 1, 128, 128}, torch::kFloat32).clone();

    torch::NoGradGuard no_grad;
    torch::Tensor output = card_model.forward({input_tensor}).toTensor();

    int pred_idx = output.argmax(1).item<int>();
    if (pred_idx < 0 || pred_idx >= static_cast<int>(card_classes.size()))
        return "Unknown";

    return card_classes[pred_idx];
}

cv::Mat extract_rank_patch_center_based(const cv::Mat &gray)
{
    const cv::Size baseWindow(70, 103);
    const double scale = 1.2;
    cv::Size winSize(cvRound(baseWindow.width * scale), cvRound(baseWindow.height * scale));

    if (gray.cols < winSize.width || gray.rows < winSize.height)
    {
        return cv::Mat::zeros(winSize, CV_8UC1);
    }

    cv::Rect win(0, 0, winSize.width, winSize.height);
    cv::Mat patch = gray(win).clone();
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(8.0, cv::Size(8, 8));
    clahe->apply(patch, patch);

    cv::Mat bin;
    cv::threshold(patch, bin, 180, 255, cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty())
    {
        return cv::Mat::zeros(winSize, CV_8UC1);
    }

    cv::Point2f center(winSize.width / 2.0f, winSize.height / 2.0f);
    double centralBoxWidth = winSize.width * 0.7;
    double centralBoxHeight = winSize.height * 0.5;
    cv::Rect centralRect(
        static_cast<int>(center.x - centralBoxWidth / 2.0),
        static_cast<int>(center.y - centralBoxHeight / 2.0),
        static_cast<int>(centralBoxWidth),
        static_cast<int>(centralBoxHeight));

    std::vector<std::pair<double, int>> validContours;

    for (size_t i = 0; i < contours.size(); i++)
    {
        const auto &contour = contours[i];

        bool touchesBorder = false;
        for (const auto &pt : contour)
        {
            if (pt.x <= 0 || pt.y <= 0 || pt.x >= bin.cols - 1 || pt.y >= bin.rows - 1)
            {
                touchesBorder = true;
                break;
            }
        }
        if (touchesBorder)
            continue;

        bool intersectsCentralRect = false;
        for (const auto &pt : contour)
        {
            if (centralRect.contains(pt))
            {
                intersectsCentralRect = true;
                break;
            }
        }
        if (!intersectsCentralRect)
            continue;

        cv::Moments m = cv::moments(contour);
        if (m.m00 == 0)
            continue;
        cv::Point2f contour_center(m.m10 / m.m00, m.m01 / m.m00);
        double dist = cv::norm(contour_center - center);
        validContours.emplace_back(dist, static_cast<int>(i));
    }

    if (validContours.empty())
    {
        return cv::Mat::ones(winSize, CV_8UC1) * 255;
    }

    std::sort(validContours.begin(), validContours.end());
    std::vector<std::vector<cv::Point>> selectedContours;
    for (size_t i = 0; i < std::min<size_t>(2, validContours.size()); ++i)
    {
        selectedContours.push_back(contours[validContours[i].second]);
    }

    cv::Mat mask = cv::Mat::zeros(patch.size(), CV_8UC1);
    cv::drawContours(mask, selectedContours, -1, 255, cv::FILLED);
    cv::Mat result(patch.size(), CV_8UC1, cv::Scalar(255));
    patch.copyTo(result, mask);
    cv::threshold(result, result, 234, 255, cv::THRESH_BINARY);

    // Optionally shift content right by adding white columns on the left
    int whiteCols = 20;
    result = add_white_columns_left(result, whiteCols);

    // cv:imshow("Extracted Rank Patch", result);
    return result;
}

cv::Mat add_white_columns_left(const cv::Mat &img, int whiteCols)
{
    if (whiteCols <= 0)
        return img;

    cv::Mat extended(img.rows, img.cols + whiteCols, CV_8UC1, cv::Scalar(255));
    img.copyTo(extended(cv::Rect(whiteCols, 0, img.cols, img.rows)));
    return extended;
}
