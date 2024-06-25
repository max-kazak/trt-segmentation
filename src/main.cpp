#include "segmentation.h"
#include "processor.h"
#include "Instrumentor.h"

#include <string>

#include <spdlog/spdlog.h>
#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>


cv::Mat loadSampleImage(const std::string& imageFile) {
    InstrumentationTimer timer("loadSampleImage");
    auto img = cv::imread(imageFile);
    if (img.empty()) {
        throw std::runtime_error("unable to read image from " + imageFile);
    }
    return img;
}


void doInference(const std::string& modelFilepath, 
                 const std::string& imageFilepath, 
                 const std::string& outPath,
                 const bool showResult) {
    InstrumentationTimer timer("main");

    auto preProcOptions = PreProcessor::Options();
    auto preProc = PreProcessor(preProcOptions);

    auto postProcOptions = PostProcessor::Options();
    auto postProc = PostProcessor(postProcOptions);

    auto engineOptions = TrtSegmentation::Options();
    auto engine = TrtSegmentation(modelFilepath, engineOptions, preProc, postProc);

    engine.initialize();

    cv::Mat inImage = loadSampleImage(imageFilepath);
    cv::Mat outClass;


    // Perform model inference on the input image
    bool err = engine.infer(inImage, outClass);

    if (!err) {
        spdlog::error("Inference failed.");
        return;
    }

    // Resize to original image
    cv::resize(outClass, outClass, inImage.size(), 0, 0, cv::INTER_NEAREST);

    // Save results to disk
    cv::imwrite(outPath + "classes.png", outClass);

    // Apply colormap
    cv::Mat colorOutput;
    cv::applyColorMap(outClass, colorOutput, cv::COLORMAP_JET);

    // Display the result
    if (showResult) {
        cv::imshow("Original image", inImage);
        cv::imshow("Segmentation Result", colorOutput);
        cv::waitKey(0); // Wait for a key press
    }
}


int main(int argcnt, char** arglist) {
    // Start profiling
    Instrumentor::Get().BeginSession("trtinfer_prof");

    // Initialize cxxopts and define options
    cxxopts::Options options("TensorRT Segmentation", 
        "This program processes input image through TensorRT model and saves resulted segmentation.");
    
    options.add_options()
        ("o,outPath", "Output path", cxxopts::value<std::string>()->default_value("../out/"))
        ("m,modelFilepath", "Model file path", cxxopts::value<std::string>()->default_value("../out/segformer-b4_opset17_fp16_trt8.trt"))
        ("i,imageFilepath", "Image file path", cxxopts::value<std::string>()->default_value("../data/img_0.png"))
        ("n,noshow", "Don't show results", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage")
    ;
    
    auto result = options.parse(argcnt, arglist);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string outPath = result["outPath"].as<std::string>();
    std::string modelFilepath = result["modelFilepath"].as<std::string>();
    std::string imageFilepath = result["imageFilepath"].as<std::string>();
    bool showResults = !result["noshow"].as<bool>();

    // Execute main program
    doInference(modelFilepath, imageFilepath, outPath, showResults);

    // End profiling
    Instrumentor::Get().EndSession();
}
