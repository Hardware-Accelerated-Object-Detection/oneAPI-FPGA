#include <sycl/sycl.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/vm.hpp>
#include <mkl.h>
#include "../../include/pipe_utils.hpp"
#include "../../include/unrolled_loop.hpp"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
using namespace sycl;

#define SampleSize 100 // the sample number in a chirp, suggesting it should be the power of 2
#define ChirpSize 128  // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize 90   // the frame number
#define RxSize 4       // the rx size, which is usually 4
#define numTx 1
#define numRx 4 // the rx size, which is usually 4
#define THREADS_PER_BLOCK 512
#define PI 3.14159265359 // pi
#define fs 2.0e6         // sampling frequency
#define lightSpeed 3.0e08
#define mu 5.987e12 // FM slope
#define f0 77e9
#define lamda lightSpeed / f0
#define d = 0.5 * lamda
#define fifo_depth 4
#define extended_length ChirpSize * ChirpSize


using preProcessingPipe = ext::intel::pipe<class preprocessingID, std::complex<double>, fifo_depth>;

using fft1dPipeArray = fpga_tools::PipeArray<class fft1dPipe, std::complex<double>, fifo_depth, RxSize, 1>;

/**
 * Host Function to submit the whole task producer 
*/
template<typename KernelClass, typename InPipe>
event SubmitHostProducer(queue &q, buffer<short,1> &base_frame, buffer<short,1> &raw_input)
{

    size_t num_elements = raw_input.size();
    std::cout << "Enqueuing Host Producer with size "<< num_elements << std::endl;
    auto e = q.submit([&](handler &h){
        accessor buf(raw_input, h, read_only);
        accessor base(base_frame, h, read_only);
        h.single_task<KernelClass>([=](){
            for(size_t i = 0; i < num_elements; i +=4)
            {   
                std::complex<double> tmp(buf[i] - base[i], buf[i+2] - base[i+2]);
                // tmp.real(real(buf[i]) + real(buf[i+2]));
                InPipe::write(tmp);
                // tmp = buf[i + 1] + buf[i+3];
                tmp.real(buf[i+1] - base[i+1]);
                tmp.imag(buf[i+3] - base[i+3]);
                // tmp.imag(imag(buf[i+1])+imag(buf[i+3]));
                InPipe::write(tmp);
            }
        });
    });
    return e;
}

/**
 * preprocessing worker: reshape / pack the complex data, feed the data into later
 * 1-D fft kernel
*/
template<typename reshapeClass, typename partitionClass,typename InPipe>
event SubmitPreProcWorker(queue &q, buffer<std::complex<double>,1> reshaped_data)
{   
    std::cout << "Enqueuing preProc Worker" << std::endl;
    size_t num_elements = reshaped_data.size();
    auto e1 = q.submit([&](handler &h){
        accessor buf(reshaped_data, h, write_only);
        h.single_task<reshapeClass>([=](){
            for(size_t srcIdx = 0; srcIdx < num_elements; srcIdx ++)
            {
                // read num_elements / 4 * 2 times
                auto data = InPipe::read();
                int chirpIdx = srcIdx / (RxSize * SampleSize);
                int rxIdx = (srcIdx - chirpIdx * RxSize * SampleSize) / SampleSize;
                int sampleIdx = srcIdx - chirpIdx * RxSize * SampleSize - rxIdx * SampleSize;
                int destIdx = rxIdx * ChirpSize * SampleSize + chirpIdx * SampleSize + sampleIdx;
                buf[destIdx] = data;
            }
        });
    });
    e1.wait();
    
    auto e = q.submit([&](handler &h){
        accessor out(reshaped_data, h, read_only);
        h.single_task<partitionClass>([=](){
            fpga_tools::UnrolledLoop<RxSize>([&](auto id){
                int start = id * SampleSize * ChirpSize;
                int end = (id + 1) * SampleSize * ChirpSize;
                // size_t num_elements = SampleSize * ChirpSize;
                for(size_t idx = start; idx < end; idx ++)
                {
                    fft1dPipeArray::PipeAt<id,0>::write(out[idx]);
                }
            });
        });
    });

    return e;
} 

SYCL_EXTERNAL int kernelBitsReverse(int num, int bits);

/**
 * submit 1d fft worker
*/
template<typename PartitionKernelClass, typename FFTKernelClass,size_t id, size_t chunk_size, size_t num_elements>
event Submit1dFFTWorker(queue &q, buffer<std::complex<double>,1> partitioned_buffer ,buffer<std::complex<double>,1> fftRes, int pow)
{
    std::cout << "Enqueing 1d FFT worker id = " << id << std::endl;
    auto e = q.submit([&](handler &h){
        accessor res(fftRes, h, write_only);
        accessor buf(partitioned_buffer,h,write_only);
        h.single_task<PartitionKernelClass>([=](){
            for(size_t i = 0; i < SampleSize * ChirpSize; i ++)
            {
                auto data =  fft1dPipeArray::PipeAt<id,0>::read();
                res[i] = data;
                buf[i] = data;
            }
        });
    });
    e.wait();
    auto fftEvent = q.submit([&](handler &h){
        accessor chunk(fftRes, h, read_write);
        h.single_task<FFTKernelClass>([=](){
            fpga_tools::UnrolledLoop<num_elements / chunk_size>([&](auto chunk_idx){
                // bit reverse for each chunk
                size_t offset = chunk_idx * chunk_size;
                // auto chunk = out[offset];
                for(size_t i = 0; i < chunk_size; i++)
                {
                    int reversedIdx = kernelBitsReverse(i, pow);
                    if(reversedIdx > i)
                    {
                        auto temp = chunk[i + offset];
                        chunk[i + offset] = chunk[reversedIdx + offset];
                        chunk[reversedIdx + offset] = temp;
                    }
                }
                // butterfly computation for each chunk 
                for(int mid = 1; mid < chunk_size; mid <<=1)
                {
                    std::complex<double> Wn(cos(PI / mid), -sin(PI/mid));
                    for(int stride = mid << 1, j = 0; j < chunk_size; j += stride)
                    {
                        std::complex<double> w(1,0);
                        for(int k = 0; k < mid; k ++, w = w * Wn)
                        {
                            auto a = chunk[j + k + offset];
                            auto b = chunk[j + mid + k + offset] * w;
                            chunk[j + k + offset] = a + b;
                            chunk[j + mid + k + offset] = a - b;
                        }
                    }
                }
            
            });

        });
    });

    return fftEvent;
}

/**
 * fft helper function
*/
namespace fft_helper{
    int bitsReverse(int num, int bits);
    void hostFFT(std::vector<std::complex<double>> &input, int bits);
}
