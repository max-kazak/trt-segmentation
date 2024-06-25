#include "segmentation.h"
#include "utils.h"
#include "Instrumentor.h"

#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>



void TrtSegmentation::initialize() {
    InstrumentationTimer timer("TrtSeg::init", true);

    setDevice();
    loadNetwork();
    createCudaStream();
    reserveMem();

    initialized_ = true;
}


void TrtSegmentation::setDevice() {
    InstrumentationTimer timer("TrtSeg::setDev");

    // set device
    spdlog::info(fmt::format(
        "setting device id to {}", options_.device_idx
        ));
    if (cudaSetDevice(options_.device_idx) != 0) {
        throw std::runtime_error("Couldn't select gpu " + options_.device_idx);
    }
}


void TrtSegmentation::loadNetwork() {  
    InstrumentationTimer timer("TrtSeg::loadNet");
    spdlog::info("Attempting to loading TensorRT engine from " + engineFile_);

    std::ifstream file(engineFile_, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open engine file " + engineFile_);
    }

    spdlog::info("reading engine file: " + engineFile_);
    // preallocate storage
    std::streamsize fsize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engineData(fsize);

    // read file
    if (!file.read(engineData.data(), fsize)) {
        throw std::runtime_error("Failed to read engine file " + engineFile_);
    }
    file.close();


    spdlog::info("creating a runtime to deserialize the engine file.");
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_)
        );
    if (!runtime_) {
        throw std::runtime_error("Failed to create Runtime");
    }

    spdlog::info("creating engine");
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engineData.data(), engineData.size())
        );
    if (!engine_) {
        throw std::runtime_error("Failed to create CudaEngine");
    }

    spdlog::info("create context");
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext()
        );
    if (!context_) {
        throw std::runtime_error("Failed to create ExecutionContext");
    }
}


void TrtSegmentation::reserveMem() {
    InstrumentationTimer timer("TrtSeg::resMem");

    for (int tensorIdx=0; tensorIdx < engine_->getNbIOTensors(); tensorIdx++) {
        const auto tensorName = engine_->getIOTensorName(tensorIdx);
        IOTensorNames_.emplace_back(tensorName);
        const auto tensorMode = engine_->getTensorIOMode(tensorName);
        const auto tensorShape = engine_->getTensorShape(tensorName);

        if (tensorMode == nvinfer1::TensorIOMode::kINPUT) {
            // input tensor
            inTensorId_ = tensorIdx;

            // save dimentions to process inputs later
            inputDims_ = nvinfer1::Dims3(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);

            //TODO: support other types of outputs
            inTensorSize_ = sizeof(float);
            for (int dimIdx=1; dimIdx < tensorShape.nbDims; dimIdx++) {
                inTensorSize_ *= tensorShape.d[dimIdx];
            }

            // allocate memory on gpu for input tensor
            utils::checkCudaErrorCode(
                cudaMallocAsync(&buffers_[tensorIdx], inTensorSize_ * options_.maxBatchSize, stream_)
            );
        } else if (tensorMode == nvinfer1::TensorIOMode::kOUTPUT){
            // output tensor
            outTensorId_ = tensorIdx;

            outputDims_ = tensorShape;
            
            //TODO: support other types of outputs
            outTensorSize_ = sizeof(float);
            for (int dimIdx=1; dimIdx < tensorShape.nbDims; dimIdx++) {
                outTensorSize_ *= tensorShape.d[dimIdx];
            }

            //allocate memory on gpu for output
            utils::checkCudaErrorCode(
                cudaMallocAsync(&buffers_[tensorIdx], outTensorSize_ * options_.maxBatchSize, stream_)
            );
        }
    }
    utils::checkCudaErrorCode(
        cudaStreamSynchronize(stream_)
    );
}


void TrtSegmentation::createCudaStream() {
    InstrumentationTimer timer("TrtSeg::createCudaStream");
    utils::checkCudaErrorCode(cudaStreamCreate(&stream_));
}


bool TrtSegmentation::infer(const cv::Mat &inputImg, cv::Mat &output) {
    InstrumentationTimer timer("TrtSeg::infer", true);

    if (!initialized_) {
        spdlog::error("TrtSegmentation has to be initialized first.");
        return false;
    }
    
    // prepare data
    cv::Mat inputData;
    preproc_.process(inputImg, inputData);
    utils::checkCudaErrorCode(
        cudaMemcpy(buffers_[inTensorId_], inputData.data, inTensorSize_, cudaMemcpyHostToDevice)
    );

    // run inference
    if (!context_->executeV2(buffers_)) {
        spdlog::error("TrtSegmentation failed at inference");
        return false;
    }

    // Retrieve output data from the GPU
    int n = outTensorSize_ / sizeof(float);
    std::vector<float> outputData(n);
    utils::checkCudaErrorCode(
        cudaMemcpy(outputData.data(), buffers_[outTensorId_], outTensorSize_, cudaMemcpyDeviceToHost)
    );

    // postprocessing
    output = postproc_.process(outputData);
    
    return true;
}


TrtSegmentation::~TrtSegmentation() {
    InstrumentationTimer timer("TrtSeg::destroy");

    cudaError_t err;

    // free buffers
    if (buffers_[0] != nullptr) {
        err = cudaFree(buffers_[0]);
        buffers_[0] = nullptr;
        if (err != cudaSuccess) 
            spdlog::error("Failed to clear buffer"); 
    }

    if (buffers_[1] != nullptr) {
        err = cudaFree(buffers_[1]);
        buffers_[1] = nullptr;
        if (err != cudaSuccess)
            spdlog::error("Failed to clear buffer");
    }

    // destroy cuda stream
    err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess)
        spdlog::error("Failed to destroy CUDA stream");

    // destroy context
    if (context_ != nullptr) {
        context_.reset();
        context_ = nullptr;
    }
    
    // destroy engine
    if (engine_ != nullptr) {
        engine_.reset();
        engine_ = nullptr;
    }

    // destroy runtime_
    if (runtime_ != nullptr) {
        runtime_.reset();
        runtime_ = nullptr;
    }
}
