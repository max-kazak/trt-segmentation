#include "processor.h"
#include "Instrumentor.h"

#include <xtensor/xsort.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>

// PreProcessor impl

void PreProcessor::process(const cv::Mat& inputImage, cv::Mat& inputData) {
    InstrumentationTimer timer("PreProc::process");
    cv::Mat img;

    // convert to rgb
    cv::cvtColor(inputImage, img, cv::COLOR_BGR2RGB);

    // resize
    cv::resize(img, img, cv::Size(options_.resize_w, options_.resize_h), 0, 0, cv::INTER_LINEAR);

    // rescale
    img.convertTo(img, CV_32FC3, options_.rescale);

    // normalize
    normalizeImage(img);

    // Convert OpenCV NHWC to TensorRT NCHW format
    cv::dnn::blobFromImage(img, inputData);
}


void PreProcessor::normalizeImage(cv::Mat& imageF) {
    InstrumentationTimer timer("PreProc::norm");

    // compute current mean and standard deviation
    cv::Scalar mean, stddev;
    cv::meanStdDev(imageF, mean, stddev);

    // split the channels
    std::vector<cv::Mat> channels;
    cv::split(imageF, channels);

    for (int ch=0; ch < 3; ch++) {
        // normalize the channel
        channels[ch] = (channels[ch] - mean[ch]) / stddev[ch];

        // scale and shift to the target mean and standard deviation
        channels[ch] = channels[ch] * options_.std[ch] + options_.mean[ch];
    }

    // Merge the channels back
    cv::merge(channels, imageF);
}


// PostProcessor impl

cv::Mat PostProcessor::process(const std::vector<float> &outData)
{
    InstrumentationTimer timer("PostProc::process");

    // convert 1D array to 3D tensor (classes x H x W)
    xt::xtensor<float, 3> outTensor = xt::adapt(outData, {options_.out_c, options_.out_h, options_.out_w});
    // reduce dimensionality to 2D by computing argmax across classes
    xt::xarray<size_t> maxIds = xt::eval(xt::argmax(outTensor, 0));
    xt::xarray<unsigned char> classIds = xt::cast<unsigned char>(maxIds);

    // convert xtensor to OpenCV
    cv::Mat outClass(options_.out_h, options_.out_w, CV_8UC1, classIds.data());

    // resize class image
    cv::resize(outClass, outClass, cv::Size(), 
                options_.resize_factor, options_.resize_factor, cv::INTER_NEAREST);


    return outClass;
}
