//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "../../include/exception_handler.hpp"

// This code sample demonstrates how to split the host and FPGA kernel code into
// separate compilation units so that they can be separately recompiled.
// Consult the README for a detailed discussion.
//  - host.cpp (this file) contains exclusively code that executes on the host.
//  - kernel.cpp contains almost exclusively code that executes on the device.
//  - kernel.hpp contains only the forward declaration of a function containing
//    the device code.
#include "kernel.hpp"

// using namespace sycl;

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
  // convert array to vector
  std::vector<short> raw_producer_input(read_data, read_data + size);
  std::vector<Complex_t> complex_consumer_output(data_per_frame / 2);
  std::vector<short> hold_consumer_internal(fifo_depth);

  event producer_event, consumer_event;
  // deploy the function into device
  try
  {
    auto props = property_list{property::queue::enable_profiling()};
    queue q(selector, fpga_tools::exception_handler, props);
    auto device = q.get_device();
    std::cout << "Runing on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;
    // allocate related buffers for devices
    buffer producer_buffer_device(raw_producer_input);
    buffer consumer_buffer_device(complex_consumer_output);
    buffer consumer_hold_buffer_device(hold_consumer_internal);
    // launch kernel function
    producer_event = ReadProducer(q, producer_buffer_device);
    consumer_event = ReshapeConsumer(q, consumer_hold_buffer_device, consumer_buffer_device);
    // syncrhonize host and device
    consumer_event.wait();
    // access the consumer output
    auto host_consumer_accessor = consumer_buffer_device.get_access<access::mode::read>();
    for (int i = 24400; i < size, i < 24430; i++)
    {
      printf("[%d] %.5f %.5f \n", i, host_consumer_accessor[i].real, host_consumer_accessor[i].imag);
    }
  }
  catch (const exception &e)
  {
    std::cerr << "Caught a SYCL host exception:\n"
              << e.what() << '\n';
    if (e.code().value() == CL_DEVICE_NOT_FOUND)
    {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
  }

  fclose(fp);
  free(read_data);
  return 0;
}
