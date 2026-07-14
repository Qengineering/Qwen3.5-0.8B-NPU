#include "RK35llm.h"
#include <vector>

int main(int argc, char** argv)
{
    // Usage: ./VLM_VIDEO_NPU vlm_model llm_model frame1.jpg [frame2.jpg frame3.jpg ...]
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " vlm_model llm_model file1.jpg [file2.jpg file3.jpg ...]\n"; 
        return -1;
    }

    std::string vlm_model = argv[1];
    std::string llm_model = argv[2];

    RK35llm RKLLM;
    RKLLM.SetInfo(true);
    RKLLM.SetSilence(false);

    RKLLM.LoadModel(vlm_model, llm_model, 2048, 16384);

    // Collect all image frames from arguments
    std::vector<cv::Mat> frames;
    for (int i = 3; i < argc; ++i) {
        cv::Mat frame = cv::imread(argv[i]);
        if (!frame.empty()) {
            frames.push_back(frame);
        } else {
            std::cerr << "Warning: Could not load image file: " << argv[i] << "\n";
        }
    }

    if (frames.empty()) {
        std::cerr << "Error: No valid images loaded. Exiting.\n";
        return -1;
    }

    // Dynamic routing: single image vs. video sequence
    if (frames.size() == 1) {
        std::cout << "\n[Info] Loading single image mode...\n";
        RKLLM.LoadImage(frames[0]);
    } else {
        std::cout << "\n[Info] Loading video sequence mode (" << frames.size() << " frames)...\n";
        RKLLM.LoadVideoFrames(frames);
    }

    std::string input_str;
    std::string output_str;

    while (true) {
        printf("\nUser: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit") break;

        // Reminder: the user must input the correct tag (<image> or <video>) based on the mode!
        output_str = RKLLM.Ask(input_str);
//        std::cout << "\nLLM Reply: " << output_str << std::endl;
    }

    return 0;
}
