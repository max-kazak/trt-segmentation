#pragma once

#include "processor.h"

#include <NvInfer.h>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>

///
/// @brief logger for TensorRT
///
class NVLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            spdlog::critical("TensorRT: {}", msg);
            break;
        case Severity::kERROR:
            spdlog::error("TensorRT: {}", msg);
            break;
        case Severity::kWARNING:
            spdlog::warn("TensorRT: {}", msg);
            break;
        case Severity::kINFO:
            spdlog::info("TensorRT: {}", msg);
            break;
        case Severity::kVERBOSE:
            spdlog::debug("TensorRT: {}", msg);
            break;
        default:
            spdlog::info("TensorRT: {}", msg);
            break;
        }
    }
};

///
/// @brief wrapper around TensorRT engine
///
class TrtSegmentation {
public:
    struct Options {
        int device_idx = 0;     // id of the gpu
        int maxBatchSize = 1;   // TODO: support batch size > 1
    };

    TrtSegmentation(const std::string& engineFile, 
                    const Options options, 
                    const PreProcessor& preproc,
                    const PostProcessor& postproc)  : 
                                 options_(options), engineFile_(engineFile), preproc_(preproc), postproc_(postproc), initialized_(false) {};
    ~TrtSegmentation();

    void initialize();
    bool infer(const cv::Mat& inputImg, cv::Mat& output);

private:
    void setDevice();
    void loadNetwork();
    void reserveMem();
    void createCudaStream();

    Options options_;
    const std::string engineFile_;

    PreProcessor preproc_;
    PostProcessor postproc_;

    bool initialized_;

    // TensorRT objects
    std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    
    // Holds pointers to the input and output GPU buffers
    void * buffers_[2] = {nullptr, nullptr};
    std::vector<std::string> IOTensorNames_;
    
    // input Tensor params
    int inTensorId_;
    nvinfer1::Dims3 inputDims_;
    uint32_t inTensorSize_;

    // output Tensor params
    int outTensorId_;
    nvinfer1::Dims outputDims_;
    uint32_t outTensorSize_;

    cudaStream_t stream_;

    NVLogger logger_;
};

