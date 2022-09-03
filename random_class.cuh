#pragma once
#include "cuda_runtime.h"
#include <cuda.h>
#include <curand.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#define errorOutCuda {std::cout << "Error in " << __FILE__ << " and " << __LINE__; goto Error;}
template <typename T>
class CudaRandomGenerator
{
private:
	static const std::string normal;
	static const std::string unifrom;
public:
	static cudaError_t generate(T* a, size_t size, const std::string method)
	{
		curandStatus_t curandomStat;
		cudaError_t cudaStat;
		curandGenerator_t gen;
		T* dev_a;
		cudaStat = cudaSetDevice(0);
		if (cudaStat == cudaSuccess)
			std::cout << "device_ini\n";
		if (cudaStat != cudaSuccess)
			errorOutCuda;
		curandomStat = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
		if (curandomStat != CURAND_STATUS_SUCCESS)
			errorOutCuda;
		curandomStat = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
		if (curandomStat != CURAND_STATUS_SUCCESS)
			errorOutCuda;
		cudaStat = cudaMalloc((void**)&dev_a, size * sizeof(T));
		if (cudaStat == cudaSuccess)
			std::cout << "malloc_ini\n";
		if (cudaStat != cudaSuccess)
			errorOutCuda;
		if (method == normal)
		{
			std::cout << "normal\n";
			curandomStat = randNormal(dev_a, size, gen);
		}
		else if (method == unifrom)
		{
			std::cout << "uniform\n";
			curandomStat = randUniform(dev_a, size, gen);
		}
		else
		{
			std::cout << "undefined\n";
			curandomStat = randNormal(dev_a, size, gen);
		}
		if (curandomStat != CURAND_STATUS_SUCCESS)
			errorOutCuda;
		cudaStat = cudaMemcpy(a, dev_a, size * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaStat != cudaSuccess)
			errorOutCuda;
	Error:
		cudaFree(dev_a);
		return cudaStat;
	}
	static inline curandStatus_t randNormal(float* dev_a, size_t size, curandGenerator_t& gen)
	{
		return 	curandGenerateNormal(gen, dev_a, size, 0.0, 1.0);
	}
	static inline curandStatus_t randNormal(double* dev_a, size_t size, curandGenerator_t& gen)
	{
		return 	curandGenerateNormalDouble(gen, dev_a, size, 0.0, 1.0);
	}
	static inline curandStatus_t randUniform(float* dev_a, size_t size, curandGenerator_t& gen)
	{
		return 	curandGenerateUniform(gen, dev_a, size);
	}
	static inline curandStatus_t randUniform(double* dev_a, size_t size, curandGenerator_t& gen)
	{
		return 	curandGenerateUniformDouble(gen, dev_a, size);
	}
};
template <typename T>
const std::string CudaRandomGenerator<T>::normal = "normal";
template <typename T>
const std::string CudaRandomGenerator<T>::unifrom = "uniform";