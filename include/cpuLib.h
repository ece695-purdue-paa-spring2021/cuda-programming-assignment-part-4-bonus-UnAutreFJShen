

#ifndef CPU_LIB_H
#define CPU_LIB_H

	#include <iostream>
	#include <cstdlib>
	#include <ctime>
	#include <random>
    #include <iomanip>
	#include <chrono>
	#include <cstring>
	#include <cstdarg>
	#include <fstream>
	#include <vector>
	#include <algorithm>
	
	// Uncomment this to suppress console output
	// #define DEBUG_PRINT_DISABLE

	extern void dbprintf(const char* fmt...);


	extern void vectorInit(float* v, int size);
	extern int verifyVector(float* a, float* b, float* c, float scale, int size);
	extern void printVector(float* v, int size);
	
	extern void saxpy_cpu(float* x, float* y, float scale, uint64_t size);

	extern int runCpuSaxpy(uint64_t vectorSize);

	extern int runCpuMCPi(uint64_t iterationCount, uint64_t sampleSize);


	////    Lab 2    ////

	typedef struct ImageDim_t
	{
		uint32_t height;
		uint32_t width;
		uint32_t channels;
		uint32_t pixelSize;
	} ImageDim;

	extern std::ostream& operator<< (std::ostream &o,ImageDim imgDim);
	
	/**
	 * @brief 
	 * 
	 * @param bytesFilePath 
	 * @param imgDim 
	 * @param imgData 
	 * @return int 
	 */
	extern int loadBytesImage(std::string bytesFilePath, ImageDim &imgDim, uint8_t ** imgData);

	extern int writeBytesImage(std::string outPath, ImageDim &imgDim, uint8_t * outData);

	typedef struct MedianFilterArgs_t {
		uint32_t filterH;
		uint32_t filterW;
	} MedianFilterArgs;

	extern int medianFilter_cpu(uint8_t inPixels, ImageDim imgDim, 
		uint8_t outPixels, MedianFilterArgs args);

	extern int runCpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args);

	enum class PoolOp{MaxPool, AvgPool, MinPool};

	extern std::ostream& operator<< (std::ostream &o,PoolOp op);

	typedef struct TensorShape_t {
		uint32_t count;		//	4th dimension	-	Quite unimaginative .. I know
		uint32_t channels;	//	3rd dimension
		uint32_t height;	//	Height = # rows	-	2nd dimension
		uint32_t width;		//	Width = # cols	-	1st dimension
	} TensorShape;

	extern std::ostream& operator << (std::ostream &o, const TensorShape & t);

	extern uint64_t tensorSize (const TensorShape & t);
	extern uint64_t tensorSize1D (const TensorShape & t);
	extern uint64_t tensorSize2D (const TensorShape & t);
	extern uint64_t tensorSize3D (const TensorShape & t);

	typedef struct PoolLayerArgs_t {
		PoolOp opType;
		uint32_t poolH;		//	pooling rows
		uint32_t poolW;		//	pooling cols
		uint32_t strideH;
		uint32_t strideW;
	} PoolLayerArgs;


	/**
	 * @brief 
	 * 
	 * @param input 
	 * @param inShape 
	 * @param output 
	 * @param outShape 
	 * @param args 
	 * @return int 
	 */
	extern int poolLayer_cpu (float * input, TensorShape inShape, 
		float * output, TensorShape outShape, PoolLayerArgs args);


	extern int runCpuPool (TensorShape inShape, PoolLayerArgs poolArgs);
	


	////    Lab 3    ////
	

	typedef struct ConvLayerArgs_t {
		uint32_t padH;
		uint32_t padW;
		uint32_t strideH;
		uint32_t strideW;
		bool activation;
	} ConvLayerArgs;

	extern int makeTensor (float ** t, TensorShape & shape);
	extern int makeVector (float ** v, uint64_t size);


	const TensorShape AlexL1_InShape 		= {1, 3, 227, 227};
	const TensorShape AlexL1_FilterShape	= {96, 3, 11, 11};
	const ConvLayerArgs AlexL1_ConvArgs 	= {0, 0, 4, 4, false};

	extern int runCpuConv (int argc, char ** argv);

	extern int executeCpuConv (TensorShape iShape, TensorShape fShape, 
		TensorShape & oShape, ConvLayerArgs args);

	#define IDX2R(r, c, cDim) ((r) * (cDim) + (c))

	/**
	 * @brief 
	 * 
	 * @param input 
	 * @param iShape 
	 * @param filter 
	 * @param fShape 
	 * @param bias 
	 * @param output 
	 * @param oShape 
	 * @param args 
	 * @param batchSize 
	 * @return int 
	 */
	extern int convLayer_cpu( float * input, TensorShape iShape, 
		float * filter, TensorShape fShape, 
		float * bias, float * output, TensorShape & oShape, 
		ConvLayerArgs & args, uint32_t batchSize);

	typedef struct GemmLayerArgs_t {
		uint32_t tileH;
		uint32_t tileW;
		uint32_t subTileCount;
	} GemmLayerArgs;

	extern int runCpuGemm (int argc, char ** argv);

	extern int executeCpuGemm (TensorShape aShape, TensorShape bShape, 
		TensorShape & cShape, GemmLayerArgs args);

	extern int gemmLayer_cpu (float * a, TensorShape aShape,
		float * b, TensorShape bShape, float * c, TensorShape & cShape,
		GemmLayerArgs & args, uint32_t batchSize = 1);

	void printTensor (float * t, TensorShape shape);

#endif
