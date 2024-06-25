#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

namespace utils {
    // Checks if a file exists at the given file path
    bool doesFileExist(const std::string &filepath);

    // Checks and logs CUDA error codes
    void checkCudaErrorCode(cudaError_t code);

    // Retrieves a list of file names in the specified directory
    std::vector<std::string> getFilesInDirectory(const std::string &dirPath);

    class ScopedTimer {
	public:
		ScopedTimer(const char* name);
		~ScopedTimer();

		void Stop();

	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> startTimepoint_;
		const char* name_;
		bool stopped_;
    };

}  //end namespace utils
