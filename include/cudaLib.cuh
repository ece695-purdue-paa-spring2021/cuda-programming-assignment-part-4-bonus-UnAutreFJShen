

#ifndef CUDA_LIB_H
#define CUDA_LIB_H

	#include "cpuLib.h"

	#include <cuda.h>
	#include <curand_kernel.h>

	// Uncomment this to suppress console output
	// #define DEBUG_PRINT_DISABLE

	//	Uncomment this to disable error counting
	//	#define CONV_CHECK_DISABLE 

	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
	extern void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

	/**
	 * @brief GPU kernel to generate sampleSize points in the 2D plane and 
	 *			evaluate the number of points that lie wihtin the unit quarter circle
	 * 
	 * @param pSums 		uint64_t	pointer to vector to store partial sum
	 *									of the number of points within unit circle
	 * @param pSumSize 		uint64_t	size of the pSum vector 
	 * @param sampleSize	uint64_t 	size of the set of random points evaluated by each GPU thread
	 * @return void 
	 */
	extern __global__ void generatePoints (uint64_t * pSums, uint64_t pSumSize, 
		uint64_t sampleSize);

	/**
	 * @brief Optional GPU kernel to reduce a set of partial sums into a smaller set
	 *			by summing a subset into a single value
	 * 
	 * @param pSums 		pointer to memory location to store partial counts
	 * @param totals 		pointer to memory location to store reduced counts
	 * @param pSumSize 		size of pSums array
	 * @param reduceSize 	size of reduced totals array
	 * @return void 
	 */
	extern __global__ void reduceCounts (uint64_t * pSums, uint64_t * totals, 
		uint64_t pSumSize, uint64_t reduceSize);

	/**
	 * @brief Entrypoint for GPU SAXPY application
	 *			- Must generate vectors of the appropriate size
	 *			- Perform GPU SAXPY
	 *			- Verify using CPU and report errors if any
	 * 
	 * @param vectorSize int	size of vector 
	 * @return int 
	 */
	extern int runGpuSaxpy(int vectorSize);

	/**
	 * @brief GPU Kernel for performing SAXPY (Y += Scale * X)
	 * 
	 * @param x 	float*		pointer to vector X
	 * @param y 	float*		pointer to vector Y
	 * @param scale float		scale factor (A of SAXPY)
	 * @param size 	uint64_t 	size of the vector Y
	 * @return void 
	 */
	extern __global__ void saxpy_gpu (float* x, float* y, float scale, int size);

	/**
	 * @brief Entrypoint for GPU Monte-Carlo estimation of Pi
	 * 
	 * @param generateThreadCount 	uint64_t	total number of generate threads	
	 * @param sampleSize 			uint64_t	sample of points evaluated by each thread
	 * @param reduceThreadCount 	uint64_t	number of reduction threads
	 * @param reduceSize 			uint64_t	number of pSums summed by each reduce thread
	 * @return int 	success or failure status
	 */
	extern int runGpuMCPi(uint64_t generateThreadCount, uint64_t sampleSize, 
		uint64_t reduceThreadCount, uint64_t reduceSize);

	/**
	 * @brief main body for Monte-Carlo Pi estimation
	 * 
	 * @param generateThreadCount 	uint64_t	total number of generate threads	
	 * @param sampleSize 			uint64_t	sample of points evaluated by each thread
	 * @param reduceThreadCount 	uint64_t	number of reduction threads
	 * @param reduceSize 			uint64_t	number of pSums summed by each reduce thread
	 * @return double 	approx value of pi
	 */
	extern double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
		uint64_t reduceThreadCount, uint64_t reduceSize);



	//	STUDENT: Add appropriate __global__ , __device__ , or __host__ keywords as applicable


	/**
	 * @brief 
	 * 
	 * @param imgPath path to the input .bytes file - to be loaded using loadBytesImage()
	 * @param outPath path to output .bytes file - to be written to using writeBytesImage()
	 * @param args settings for median filter
	 * @return int 
	 */
	extern int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args);


	/**
	 * @brief GPU kernel 
	 * 
	 * @param inPixels 	uint8_t *			pointer to input of medianFilter
	 * @param imgDim 	ImageDim			struct converying the diemnsionality of inPixels
	 * @param outPixels uint8_t *			pointer to output of medianFilter
	 * @param args 		MedianFilterArgs	struct with settings for median filter
	 * @return int 
	 */
	extern int medianFilter_gpu (uint8_t * inPixels, ImageDim imgDim, 
		uint8_t * outPixels, MedianFilterArgs args);

	/**
	 * @brief GPU kernel to sort a given array
	 * 
	 * @param array 	uint8_t *	pointer to the array to be sorted 
	 *								(Can point to anywhere within the addressable memory heirarchy)
	 * @param arrayDim 	dim3		provides upto 3 dimensions to convey how the data chunk 
	 *								pointed to by array is organized
	 * @return int 
	 */
	extern int sortPixels_gpu (uint8_t * array, dim3 arrayDim);

	/**
	 * @brief CPU entrypoint for GPU based pool operation
	 *			- allocate required memory on host and device
	 *			- execute gpu kernel to pool
	 *			- verify pooling - report # errors
	 * 
	 * @param inShape 	dimensions of input tensor
	 * @param poolArgs 	PoolLayerArgs	parameters of pool operation
	 * @return int 		number of errors in pooled output
	 */
	extern int runGpuPool (TensorShape inShape, PoolLayerArgs poolArgs);

	/**
	 * @brief GPU kernel to perform 2D Pool operation
	 * 
	 * @param input 	float *			pointer to input tensor
	 * @param inShape 	TensorShape		dimensions of input tensor
	 * @param output 	float *			pointer to output tensor
	 * @param outShape 	TensorShape		dimensions of output tensor
	 * @param args 		PoolLayerArgs	parameters of pool operation
	 * @return int 
	 */
	extern int poolLayer_gpu (float * input, TensorShape inShape,
		float * output, TensorShape outShape, PoolLayerArgs args);

	/**
	 * @brief 
	 * 
	 * @param argc 
	 * @param argv 
	 * @return int 
	 */
	extern int runGpuConv (int argc, char ** argv);

	/**
	 * @brief 
	 * 
	 * @param iShape 	TensorShape
	 * @param fShape 	TensorShape
	 * @param oShape 	TensorShape &		output tensor dimensions - reference
	 * @param args 		ConvLayerArgs
	 * @return 			uint64_t	 	number of errors
	 */
	extern uint64_t evaluateGpuConv (TensorShape iShape, TensorShape fShape, 
		TensorShape & oShape, ConvLayerArgs args);
	
	
	/**
	 * @brief 
	 * 
	 * @param input 	float *
	 * @param iShape 	TensorShape
	 * @param filter 	float *
	 * @param fShape 	TensorShape
	 * @param bias 		float *
	 * @param output 	float *
	 * @param oShape 	TensorShape		dimensions of output tensor
	 * @param args 		ConvLayerArgs	parameters for convolution operation
	 * @param batchSize uint32_t		
	 * @return int 
	 */
	extern int convLayer_gpu ( float * input, TensorShape iShape, 
		float * filter, TensorShape fShape, 
		float * bias, float * output, TensorShape & oShape, 
		ConvLayerArgs & args, uint32_t batchSize);

	extern int runGpuGemm (int argc, char ** argv);

	extern int gemmLayer_gpu (float * a, TensorShape aShape, 
		float * b, TensorShape bShape,
		float * c, TensorShape & cShape,
		GemmLayerArgs & args, uint32_t batchSize);

	extern int runGpuGemm (int argc, char ** argv);

	extern int evaluateGpuGemm ();

#endif
