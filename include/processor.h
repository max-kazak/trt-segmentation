#pragma once

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"

///
/// @brief Prepares input images for the model inference.
///
class PreProcessor {
public:
    struct Options {
        int resize_w = 512;                             // width of the input tensor
        int resize_h = 512;                             // height
        double rescale = 1.0 / 255.0;                   // value rescaling factor
        std::vector<float> mean = {.485, .456, .406};   // target mean of the channels
        std::vector<float> std = {.229, .224, .225};    // target std
    };

    PreProcessor(Options options) : options_(options) {};

    void process(const cv::Mat& inputImage, cv::Mat& inputData);

private:
    void normalizeImage(cv::Mat& image);

    Options options_;
};

///
/// @brief Processes model output
///
class PostProcessor {
public:
    struct Options {
        int out_h = 128;            // model output tensor height
        int out_w = 128;            // width
        int out_c = 150;            // number of classes
        double resize_factor = 4;   // resizing factor
    };

    PostProcessor(Options options) : options_(options) {};

    cv::Mat process(const std::vector<float>& outData);

private:
    Options options_;
};
