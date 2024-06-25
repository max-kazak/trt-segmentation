#include "utils.h"

#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>

namespace utils {

bool doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

void checkCudaErrorCode(cudaError_t code) {
    if (code != cudaSuccess) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + " (" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        spdlog::error(errMsg);
        throw std::runtime_error(errMsg);
    }
}

std::vector<std::string> getFilesInDirectory(const std::string &dirPath) {
    std::vector<std::string> fileNames;
    for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            fileNames.push_back(entry.path().string());
        }
    }
    return fileNames;
}

// Timers

ScopedTimer::ScopedTimer(const char* name) : name_(name), stopped_(false){
	startTimepoint_ = std::chrono::high_resolution_clock::now();
}

ScopedTimer::~ScopedTimer(){
    if (!stopped_)
        Stop();
}
		
void ScopedTimer::Stop(){
    auto endTimepoint = std::chrono::high_resolution_clock::now();
    
    auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTimepoint_).time_since_epoch().count();
    auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();
    
    double duration = end - start;

    // output ms
    spdlog::info(fmt::format("Timer: {} time: {}us", name_, duration));
    
    stopped_ = true;
}


} //end namespace utils
