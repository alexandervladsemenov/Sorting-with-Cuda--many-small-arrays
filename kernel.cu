#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <stdio.h>
#include "random_class.cuh"
#include "Sorting_implementation.cuh"
#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include "sorting_cpu.h"
#define print(x) (std::cout<<x)
#define printArray(x,l) for(int i=0;i<l;i++){ print(x[i]);\
print(' ');} print('\n')


#define IDX2C(r,c,cols) r*cols + c
#define timeNow std::chrono::high_resolution_clock::now()
#define timeUnit std::chrono::high_resolution_clock::time_point
#define castTime(x) std::chrono::duration_cast<std::chrono::milliseconds>(x).count()
template <class T>
__device__ func_sort<T> dev_func_ptr = selectionSort<T>;
// data 

constexpr auto numThreads = 32 * 4;
constexpr auto sharedBlockSize = numThreads * 4;
const dim3 dimBlock(numThreads, 1, 1);
int device;
cudaError_t cudaStat;
// prepare the block
dim3 dimGridCalc(dim3 dimBlockInp, size_t rows, size_t columns = 1, size_t width = 1)
{
	auto lx = dimBlockInp.x;
	auto ly = dimBlockInp.y;
	auto lz = dimBlockInp.z;
	size_t dimxG = (rows + lx - 1) / lx;
	size_t dimyG = (columns + ly - 1) / ly;
	size_t dimzG = (width + lz - 1) / lz;
	return dim3(dimxG, dimyG, dimzG);;
	// call kernel <<<dimGrid,dimBlock>>>(...)
}

template<typename T>
void cpu_sort(T* data, int number_of_arrays, int size_of_array)
{
	for (int i = 0; i < number_of_arrays; i++)
	{
		int globalIndex = IDX2C(i, 0, size_of_array);
		T* start = &data[globalIndex];
		std::sort(start, start + size_of_array);
	}

}

template<typename T, int num>

void cpu_sort(T* data, int number_of_arrays, int size_of_array)
{
	func_sort<T> fn;
	switch (num)
	{
	default:
		fn = cpu_sort_lib::quickSort<T>;
		break;
	case 1:
		fn = cpu_sort_lib::heapSort<T>;
		break;
	case 2:
		fn = cpu_sort_lib::insertionSort<T>;
		break;
	case 3:
		fn = cpu_sort_lib::bubbleSort<T>;
		break;
	case 4:
		fn = cpu_sort_lib::quickSort<T>;
		break;
	case 5:
		fn = cpu_sort_lib::shellSort<T>;
		break;
	case 6:
		fn = cpu_sort_lib::selectionSort<T>;
		break;
	}

	for (int i = 0; i < number_of_arrays; i++)
	{
		int globalIndex = IDX2C(i, 0, size_of_array);
		T* start = &data[globalIndex];
		fn(start, size_of_array);
	}
}

template<typename T>
bool cpu_compare(T* data1, T* data2,int size)
{
	for (int i = 0; i < size; i++)
	{
		if (data1[i] != data2[i])
		{
			std::cout << "The wrong point is " << i << " wih vals " << data1[i] << " and " << data2[i]<< std::endl;
			return false;
		}
	}
	return true;
}
template<typename T, int num>
// one thread sorts all small arrays
__global__ void custom_sorting_simple(T* input, int number_of_arrays, int size_of_array)
{
	unsigned int r = blockIdx.x * blockDim.x + threadIdx.x; // index r goes of number of arrays, from 0 to number_of_arrays-1
	unsigned int globalIndex = IDX2C(r, 0, size_of_array);
	func_sort<T> fn;
	switch (num)
	{
	default:
		fn = quickSort<T>;
		break;
	case 1:
		fn = heapSort<T>;
		break;
	case 2:
		fn = insertionSort<T>;
		break;
	case 3:
		fn = bubbleSort<T>;
		break;
	case 4:
		fn = quickSort<T>;
		break;
	case 5:
		fn = shellSort<T>;
		break;
	case 6:
		fn = selectionSort<T>;
		break;
	}
	if (r < number_of_arrays)
	{
		//printf("%d %d \n", globalIndex, size_of_array);
		T* start = &input[globalIndex];
		//quickSort(start, size_of_array);
		//insertionSort(start, size_of_array);
		//shellSort(start, size_of_array);
		//heapSort(start, size_of_array);
		//bubbleSort(start, size_of_array);
		fn(start, size_of_array);

	}

}



template<class T, int case_funct>
becnhmark test(int number_of_arrays, int size_of_array)
{
	becnhmark data_timing;
	// setting the sorting algorithn
	// allocating memory
	size_t size = number_of_arrays * size_of_array;
	// creating pointers to host, device arrays
	float* a, * dev_a;
	a = new float[size];
	size_t bytes = sizeof(*a) * size;
	print("GB of memory the array takes \n");
	print(((float)bytes) / 1024 / 1024 / 1024);
	print("\n");
	// gpu memeory
	size_t mem_tot_gpy, mem_free_gpu;
	cudaMemGetInfo(&mem_tot_gpy, &mem_free_gpu);
	print("\nGB of memory in the card\n");
	print(((float)mem_tot_gpy) / 1024 / 1024 / 1024);
	print("\n\n");
	// filling up bost host and device arrays

	{	//device events
		cudaEvent_t start, stop;
		float milliseconds = 0;

		// measure time with cuda

		cudaStream_t s0;
		cudaStreamCreate(&s0);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, s0);
		cudaEventSynchronize(start);
		// timers have been set. Now ini arrays
		CudaRandomGenerator<float>::generate(a, size, "normal");
		// copying
		cudaMalloc((void**)&dev_a, size * sizeof(*a));
		cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice);
		cudaEventRecord(stop, s0); // record the stop event
		cudaEventSynchronize(stop);// blocks cpu exec
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaStreamDestroy(s0);
		cudaDeviceSynchronize();
		print("\ntimeElapsed in GPU, nilliseconds\n");
		print(milliseconds);
		data_timing.time_gpu_spent_allocation = milliseconds;
	}
	{
		print("\n Several vals of the array \n");
		printArray(a, 10);
		print("\n");

	}
	// CPU sort
	//if (false)
	{
		timeUnit t1 = timeNow;
		cpu_sort<float,case_funct>(a, number_of_arrays, size_of_array);
		timeUnit t2 = timeNow;
		print("\ntimeElapsed in CPU sorting measured by CPU, nilliseconds\n");
		print(castTime(t2 - t1));
		data_timing.time_cpu_spent_sorting = float(castTime(t2 - t1));
	}
	{
		/// <summary>
		/// setting timers
		/// </summary>
		/// <param name="argc"></param>
		/// <param name="argv"></param>
		/// <returns></returns>
		cudaEvent_t _start, _stop;
		cudaStream_t s0; // cudaStream must be refreshed?
		cudaStreamCreate(&s0);
		cudaEventCreate(&_start);
		cudaEventCreate(&_stop);
		cudaEventRecord(_start, s0);
		float milliseconds = 0.0;
		timeUnit t1 = timeNow;
		/// <summary>
		///  setting dims for kernel
		/// </summary>
		/// <param name="argc"></param>
		/// <param name="argv"></param>
		/// <returns></returns>
		/// 
		dim3 dimGrid = dimGridCalc(dimBlock, number_of_arrays);
		// start the kernel
		cudaEventSynchronize(_start);
		custom_sorting_simple<T, case_funct> << <dimGrid, dimBlock >> > (dev_a, number_of_arrays, size_of_array);
		// stop recordng
		cudaEventRecord(_stop, s0); // record the stop event
		cudaEventSynchronize(_stop);// blocks cpu exec
		// sync timers
		cudaDeviceSynchronize();
		timeUnit t2 = timeNow;
		// how much time it took
		cudaEventElapsedTime(&milliseconds, _start, _stop);
		print("\ntimeElapsed in GPU sorting, nilliseconds\n");
		print(milliseconds);
		cudaEventDestroy(_start);
		cudaEventDestroy(_stop);
		cudaStreamDestroy(s0);
		print("\ntimeElapsed in GPU sorting measured by CPU, nilliseconds\n");
		print(castTime(t2 - t1));
		data_timing.time_gpu_spent_sorting = milliseconds;
		//
	}
	{
		float* b = new float[size];
		cudaStat = cudaMemcpy(b, dev_a, size * sizeof(*b), cudaMemcpyDeviceToHost);
		if (cudaStat != cudaSuccess)
			print("\nError copying back to host\n");
		if (cpu_compare(a, b, size))
			print("\nCorrectly sorted by GPU\n");
		else
		{
			print("\nWrong sorting\n");
		}
		delete[] a;
		delete[] b;
		cudaFree(dev_a);
	}
	return data_timing;
}

void write(becnhmark test, const std::string & output_folder, const std::string& name, int number_of_arrays, int size_of_array)
{
	{

		const std::string path = output_folder + name;
		std::ofstream ofs(path, std::ios_base::app);
		ofs << number_of_arrays << " " << size_of_array << std::endl;
		ofs << test.time_cpu_spent_sorting << " " << test.time_gpu_spent_allocation << " " << test.time_gpu_spent_sorting << std::endl;
		ofs.close();

	}
}

int main(int argc, char* argv[])

{
	if (argc < 4)
		return EXIT_FAILURE;
	// preparing device and arrays
	int number_of_arrays = atoi(argv[1]);
	int size_of_array = atoi(argv[2]);
	std::string output_folder = std::string(argv[3]);
	int device_Count;
	cudaGetDeviceCount(&device_Count);
	printf("\n\nDevice Numbers: %d\n\n", device_Count);
	device = device_Count - 1;
	auto status = cudaSetDevice(device);

	if (status != cudaSuccess) {

		printf("!!!!  Set Device error\n");

		return EXIT_FAILURE;

	}

	auto time_set_1 = test<float, 1>(number_of_arrays, size_of_array);
	auto time_set_2 = test<float, 2>(number_of_arrays, size_of_array);
	auto time_set_3 = test<float, 3>(number_of_arrays, size_of_array);
	auto time_set_4 = test<float, 4>(number_of_arrays, size_of_array);
	auto time_set_5 = test<float, 5>(number_of_arrays, size_of_array);
	auto time_set_6 = test<float, 6>(number_of_arrays, size_of_array);
	cudaDeviceReset();
	write(time_set_1, output_folder, "//heapsort.txt", number_of_arrays, size_of_array);
	write(time_set_2, output_folder, "//insertionSort.txt", number_of_arrays, size_of_array);
	write(time_set_3, output_folder, "//bubbleSort.txt", number_of_arrays, size_of_array);
	write(time_set_4, output_folder, "//quickSort.txt", number_of_arrays, size_of_array);
	write(time_set_5, output_folder, "//shellSort.txt", number_of_arrays, size_of_array);
	write(time_set_6, output_folder, "//selectionSort.txt", number_of_arrays, size_of_array);
	return 0;
}


