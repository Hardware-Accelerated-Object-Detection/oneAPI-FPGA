//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "oneapi/mkl.hpp"
#include "../../include/exception_handler.hpp"
#include <complex>
#include <mkl.h>
// This code sample demonstrates how to split the host and FPGA kernel code into
// separate compilation units so that they can be separately recompiled.
// Consult the README for a detailed discussion.
//  - host.cpp (this file) contains exclusively code that executes on the host.
//  - kernel.cpp contains almost exclusively code that executes on the device.
//  - kernel.hpp contains only the forward declaration of a function containing
//    the device code.
#include "kernel.hpp"

// using namespace sycl;

static inline int nextPow2(int n)
{
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

bool partitionVerification(std::array<std::vector<std::complex<double>>, RxSize> toComp, std::vector<std::complex<double>> ref)
{
  for (int i = 0; i < 1; i++)
  {
    // printf("Rx[%d] size %d \n", i, partitioned_data[i].size());
    for (int j = 0; j < SampleSize * ChirpSize; j++)
    {
      printf("data[%d] (%.3f %.3f) \n",j, real(toComp[i][j]),imag(toComp[i][j]));
      int globalIdx = i * SampleSize * ChirpSize + j;
      double tolerance = 1e-5;
      std::complex<double> diff = toComp[i][j] - ref[globalIdx];
      if (abs(imag(diff)) > tolerance || abs(real(diff)) > tolerance)
      {
        printf("Miss match at rx[%d][%d] vs. reshaped[%d] \n", i, j, globalIdx);
        printf("input: %.3f %.3f expected %.3f %.3f \n", real(toComp[i][j]), imag(toComp[i][j]), real(ref[globalIdx]), imag(ref[globalIdx]));
        return false;
      }
    }
  }
  std::cout << "partiton Verification Passed\n"
            << std::endl;
  return true;
}

bool vectorVerification(std::vector<std::complex<double>> fftRes, std::vector<std::complex<double>> &ref, int idx)
{
  std::cout << "Verify vector for " << idx << std::endl;
  double tolerance = 1e-5;
  // doing fft for every ref input
  // for (int i = 0; i < SampleSize * ChirpSize; i++)
  for (int i = 0; i < 10; i++)

  {
    printf("fftRes[%d] (%.3f  %.3f) ref[%d] (%.3f %.3f)\n",i, real(fftRes[i]), imag(fftRes[i]), i, real(ref[i]), imag(ref[i]));
    // printf("fftRes[%d] (%.3f  %.3f)\n",i, real(fftRes[i]), imag(fftRes[i]));
    std::complex<double> diff = fftRes[i] - ref[i];
    if (abs(imag(diff)) > tolerance || abs(real(diff)) > tolerance)
    {
      printf("Verification failed at %d input: (%.3f %.3f) expected (%.3f %.3f)\n", i, real(fftRes[i]), imag(fftRes[i]), real(ref[i]), imag(ref[i]));
      return false;
    }
  }
  printf("Vector Verification passed\n");
  return true;
}

int main(int argc, char *argv[])
{
  // Select either the FPGA emulator, FPGA simulator or FPGA device
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif


  auto props = property_list{property::queue::enable_profiling()};
      queue q(selector, fpga_tools::exception_handler, props);
      auto device = q.get_device();
      std::cout << "Runing on device: "
                << device.get_info<sycl::info::device::name>().c_str()
                << std::endl;

  std::cout << "Reading raw data from binary file \n";
  char filepath[] = "../src/fhy_direct.bin";
  int data_per_frame = ChirpSize * SampleSize * numRx * 2;
  int byte_per_frame = data_per_frame * sizeof(short);
  FILE *fp = fopen(filepath, "rb");
  if (fp == NULL)
  {
    std::cout << "cannot open file " << filepath << std::endl;
  }

  short *read_data = (short *)malloc(byte_per_frame);
  int size = 0;
  int frameCnt = 0;

  size = (int)fread(read_data, sizeof(short), data_per_frame, fp);
  frameCnt++;
  // allocate the base frame
  std::vector<short> base_frame(read_data, read_data + size);
  std::array<std::vector<std::complex<double>>, RxSize> partitioned_data;
  std::array<std::vector<std::complex<double>>, RxSize> fft_res;
  std::vector<short> raw_producer_input(size);
  std::vector<std::complex<double>> reshaped_data(data_per_frame / 2);
  std::vector<short> hold_consumer_internal(fifo_depth);
  
  // while ((size = (int)fread(read_data, sizeof(short), data_per_frame, fp)) > 0)
  // { 
    size = (int)fread(read_data, sizeof(short), data_per_frame, fp);
    frameCnt ++;
    // try
    // {
      /**
       * preprocessing
       */
      raw_producer_input.insert(raw_producer_input.begin(), read_data, read_data+size);
      
      // allocate related buffers for devices
      buffer base_frame_buffer_device(base_frame);
      buffer producer_buffer_device(raw_producer_input);
      buffer consumer_buffer_device(reshaped_data);
      buffer consumer_hold_buffer_device(hold_consumer_internal);
      // launch kernel function
      std::cout << "Process base frame data" << std::endl;
      auto reshape_producer_event = ReadProducer(q, producer_buffer_device, base_frame_buffer_device);
      auto reshape_consumer_event = ReshapeConsumer(q, consumer_hold_buffer_device, consumer_buffer_device);
      /**
       * Parition
       */

      // constexpr int rx_extend_size = nextPow2(SampleSize * ChirpSize);
      constexpr int rx_extend_size = ChirpSize * ChirpSize;

      for (int i = 0; i < RxSize; i++)
      {
        partitioned_data[i].resize(rx_extend_size - SampleSize * ChirpSize, 0);
        fft_res[i].resize(rx_extend_size, 0);
      }
      int cnt = 1, pow = 0;
      while (cnt < rx_extend_size)
      {
        cnt <<= 1;
        pow++;
      }

      reshape_consumer_event.wait();
      // partition the complete data into seperate vector
      // auto partitionEvent = q.submit([&] (handler &h) {
        fpga_tools::UnrolledLoop<RxSize>([&](auto i){
          // parallely partition the data
          auto it = partitioned_data[i].begin();
          int start = i * SampleSize * ChirpSize;
          int end = (i + 1) * SampleSize * ChirpSize;
          partitioned_data[i].insert(it, reshaped_data.begin() + start, reshaped_data.begin()+end); 
        });
      // });

      // verify the partitioned data
      // partitionEvent.wait();
      assert(partitionVerification(partitioned_data, reshaped_data));

      /**
       * Distance Detection
       */
      constexpr size_t chunk_size = rx_extend_size;
      constexpr size_t num_elements = rx_extend_size;
      auto fftEvent = q.submit([&](handler &h){ 
        fpga_tools::UnrolledLoop<RxSize>([&](auto id){
          // lauch fftProducer
          // constexpr size_t id = 0;
          buffer<std::complex<double>, 1> fft_input(partitioned_data[id]);
          auto fftProducerEvent = fftProducer<id>(q, fft_input);
          buffer<std::complex<double>, 1> fft_output(fft_res[id]);
          auto fftConsumerEvent = fftConsumer<id, chunk_size, num_elements>(q, fft_output, pow); 
        }); 
      });
      // fft verification
      fftEvent.wait();
      for (int i = 0; i < 1; i++)
      {
        std::vector<std::complex<double>> buf(partitioned_data[i].begin(), partitioned_data[i].end());
        // printf("partitioned_data[%d] size %lu buf size %lu\n", i, partitioned_data[i].size(), buf.size());
        fft_helper::hostFFT(buf, pow);
        assert(vectorVerification(fft_res[i], buf, i));
      }
        printf("fft check finished\n");
      // find the max index

      /**
       * Speed Detection
       */
        printf("speed detection\n");

      /**
       * Angle Detection
       */
    // }
    // catch (const exception &e)
    // {
    //   std::cerr << "Caught a SYCL host exception:\n"
    //             << e.what() << '\n';
    //   if (e.code().value() == CL_DEVICE_NOT_FOUND)
    //   {
    //     std::cerr << "If you are targeting an FPGA, please ensure that your "
    //                  "system has a correctly configured FPGA board.\n";
    //     std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
    //     std::cerr << "If you are targeting the FPGA emulator, compile with "
    //                  "-DFPGA_EMULATOR.\n";
    //   }
    // }
  

  // }
    fclose(fp);
    free(read_data);
  return 0;
}
