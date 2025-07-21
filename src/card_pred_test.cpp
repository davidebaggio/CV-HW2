#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // ======== MODELLO ========
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("simple_card_classifier_traced.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Errore nel caricamento del modello: " << e.what() << std::endl;
        return -1;
    }
    model.eval();

    // ======== CLASSI ========
    std::vector<std::string> classes = {"10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"};

    // ======== IMMAGINE ========
    std::string image_path = "test_image.png"; // Sostituisci con percorso reale
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Errore: immagine non caricata." << std::endl;
        return -1;
    }

    // Resize 128x128
    cv::resize(image, image, cv::Size(128, 128));

    // Convert to float e normalizza [-1, 1]
    image.convertTo(image, CV_32F, 1.0 / 255);
    image = (image - 0.5) / 0.5;

    // ======== PREPROCESSING ========
    // [1, 1, 128, 128] tensor (batch, channels, height, width)
    torch::Tensor input_tensor = torch::from_blob(image.data, {1, 1, 128, 128}, torch::kFloat32).clone();

    // ======== INFERENZA ========
    torch::NoGradGuard no_grad;
    torch::Tensor output = model.forward({input_tensor}).toTensor();

    auto pred_class = output.argmax(1).item<int>();
    std::cout << "Predizione: " << classes[pred_class] << std::endl;

    return 0;
}
