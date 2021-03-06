/**
 * @file lab3.cuh
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-01-16
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once

#ifndef LAB3_CUH
#define LAB3_CUH

	#include "cpuLib.h"
	// #define DEBUG_PRINT_DISABLE
	
	#define VECTOR_SIZE (1 << 15)

	#define MC_SAMPLE_SIZE		1e6
	#define MC_ITER_COUNT		32

	#define WARP_SIZE			32
	#define SAMPLE_SIZE			MC_SAMPLE_SIZE
	#define GENERATE_BLOCKS		1024
	#define REDUCE_SIZE			32
	#define REDUCE_BLOCKS		(GENERATE_BLOCKS / REDUCE_SIZE)

	extern int testLoadBytesImage(std::string filePath);


#endif