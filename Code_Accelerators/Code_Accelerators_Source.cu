#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <benchmark/benchmark.h>

//Created by Chase Morgan

using namespace std;

//from: https://github.com/hsaputra/cuda_pi_montecarlo/blob/master/pi.cu
// Create a kernel to estimate pi
__global__
void count_samples_in_circles(float* d_randNumsX, float* d_randNumsY, int* d_countInBlocks, int num_blocks, int nsamples)
{

    __shared__ int shared_blocks[500];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * num_blocks;

    // Iterates through
    int inCircle = 0;
    for (int i = index; i < nsamples; i += stride) {
        float xValue = d_randNumsX[i];
        float yValue = d_randNumsY[i];

        if (xValue * xValue + yValue * yValue <= 1.0f) {
            inCircle++;
        }
    }

    shared_blocks[threadIdx.x] = inCircle;

    __syncthreads();

    // Pick thread 0 for each block to collect all points from each Thread.
    if (threadIdx.x == 0)
    {
        int totalInCircleForABlock = 0;
        for (int j = 0; j < blockDim.x; j++)
        {
            totalInCircleForABlock += shared_blocks[j];
        }
        d_countInBlocks[blockIdx.x] = totalInCircleForABlock;
    }
}

//Created to mimic the GPU variant as much as possible on the CPU itself.
void count_samples_in_circles_CPU(float* d_randNumsX, float* d_randNumsY, int* d_countInBlocks, int num_blocks, int nsamples)
{
    const int shared_blocks = 500;

    for (int blockIdx = 0; blockIdx < num_blocks; blockIdx++)
    {
        int totalInCircleForABlock = 0;

        for (int threadIdx = 0; threadIdx < shared_blocks; threadIdx++)
        {
            int inCircle = 0;
            //Trying to make this as comparable to the GPU code, but I personally like index instead of i for iterators.
            for (int index = 0; index < nsamples; index += (shared_blocks * num_blocks))
            {
                float xValue = d_randNumsX[index];
                float yValue = d_randNumsY[index];

                if (xValue * xValue + yValue * yValue <= 1.0f) 
                {
                    inCircle++;
                }
            }

            totalInCircleForABlock += inCircle;
        }

        d_countInBlocks[blockIdx] = totalInCircleForABlock;
    }
}

static void BM_MonteCarloPiCUDA(benchmark::State& state)
{
    const std::int64_t nsamples = state.range(0);
    float* data;
    cudaMalloc(&data, state.range(0) * sizeof(float));

    // allocate space to hold random values    
    float* h_randNumsX, *h_randNumsY;

    cudaHostAlloc(&h_randNumsX, nsamples * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_randNumsY, nsamples * sizeof(float), cudaHostAllocDefault);

    srand(time(NULL)); // seed with system clock

    //Initialize vector with random values    
    for (int i = 0; i < nsamples; ++i)
    {
        h_randNumsX[i] = float(rand()) / RAND_MAX;
        h_randNumsY[i] = float(rand()) / RAND_MAX;
    }

    // Send random values to the GPU    
    size_t size = nsamples * sizeof(float);
    float* d_randNumsX;
    float* d_randNumsY;

    cudaMalloc(&d_randNumsX, size);
    cudaMalloc(&d_randNumsY, size);


    // Launch kernel to count samples that fell inside unit circle    
    int threadsPerBlock = 500;
    int num_blocks = nsamples / (1000 * threadsPerBlock);

    //This is to prevent pi being zero.
    if (num_blocks < 1) num_blocks = 1;

    size_t countBlocks = num_blocks * sizeof(int);


    int* d_countInBlocks;
    cudaMalloc(&d_countInBlocks, countBlocks);


    for (auto _ : state)
    {
        cudaMemcpy(d_randNumsX, h_randNumsX, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_randNumsY, h_randNumsY, size, cudaMemcpyHostToDevice);
        count_samples_in_circles << <num_blocks, threadsPerBlock >> > (d_randNumsX, d_randNumsY, d_countInBlocks, num_blocks, nsamples);
        cudaDeviceSynchronize();
    }

    // Return back the vector from device to host
    int* h_countInBlocks = new int[num_blocks];
    cudaMemcpy(h_countInBlocks, d_countInBlocks, countBlocks, cudaMemcpyDeviceToHost);

    int nsamples_in_circle = 0;
    for (int i = 0; i < num_blocks; i++) {
        //cout << "Value in block " + i << " is " << h_countInBlocks[i] << endl;
        nsamples_in_circle = nsamples_in_circle + h_countInBlocks[i];
    }

    cudaFreeHost(h_randNumsX);
    cudaFreeHost(h_randNumsY);
    cudaFree(d_randNumsX);
    cudaFree(d_randNumsY);
    cudaFree(d_countInBlocks);

    // fraction that fell within (quarter) of unit circle
    float estimatedValue = 4.0 * float(nsamples_in_circle) / nsamples;

    //Used to be a cout, but listing it on the actual benchmark makes it a tiny bit clearer.
    state.counters["Pi"] = estimatedValue;

    cudaFree(data);
}

static void BM_MonteCarloPiCPU(benchmark::State& state)
{
    const std::int64_t nsamples = state.range(0);
   
    std::vector<float> h_randNumsX(nsamples);
    std::vector<float> h_randNumsY(nsamples);
    srand(time(NULL));

    for (int index = 0; index < nsamples; ++index)
    {
        h_randNumsX[index] = float(rand()) / RAND_MAX;
        h_randNumsY[index] = float(rand()) / RAND_MAX;
    }


    int threadsPerBlock = 500;
    int num_blocks = nsamples / (1000 * threadsPerBlock);

    if (num_blocks < 1) num_blocks = 1;

    std::vector<int> h_countInBlocks(num_blocks, 0);

    for (auto _ : state)
    {
        count_samples_in_circles_CPU(h_randNumsX.data(), h_randNumsY.data(), h_countInBlocks.data(), num_blocks, nsamples);
        benchmark::DoNotOptimize(h_countInBlocks.data());
        benchmark::ClobberMemory();
    }

    int nsamples_in_circle = 0;
    for (int i = 0; i < num_blocks; i++) {
        nsamples_in_circle += h_countInBlocks[i];
    }

    float estimatedValue = 4.0 * float(nsamples_in_circle) / nsamples;

    state.counters["Pi"] = estimatedValue;
}

const double zC = 2.78; //99% confidence.
const double sigma = 0.05; //0.05 second deviation.
const double E = 0.01; //0.01 seconds margin of error.

int iterations = static_cast<int>(pow((zC * sigma) / E, 2));

BENCHMARK(BM_MonteCarloPiCUDA)->Iterations(iterations)->RangeMultiplier(2)->Range(1024, 134217728)->Repetitions(25);
BENCHMARK(BM_MonteCarloPiCPU)->Iterations(iterations)->RangeMultiplier(2)->Range(1024, 134217728)->Repetitions(25);
BENCHMARK_MAIN();
