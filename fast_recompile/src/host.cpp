#include "../../include/exception_handler.hpp"
#include "kernel.hpp"
#include "lib.hpp"
#include <complex>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
static inline int nextPow2(int n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

int length2Pow(int len) {
  int cnt = 0;
  int tmp = 1;
  while (tmp < len) {
    tmp <<= 1;
    cnt++;
  }
  return cnt;
}

int findAbsMax(std::vector<std::complex<float>> &arr, int size) {
  int maxIdx = 0;
  for (int i = 0; i < size; i++) {

    if (std::abs(arr[i]) > std::abs(arr[maxIdx])) {
      maxIdx = i;
    }
  }
  return maxIdx;
}

bool partitionVerification(
    std::array<std::vector<std::complex<float>>, RxSize> toComp,
    std::vector<std::complex<float>> ref) {
  for (int i = 0; i < RxSize; i++) {
    // printf("Rx[%d] size %d \n", i, partitioned_data[i].size());
    for (int j = 0; j < SampleSize * ChirpSize; j++) {
      // printf("data[%d] (%.3f %.3f) \n", j, real(toComp[i][j]),
      // imag(toComp[i][j]));
      int globalIdx = i * SampleSize * ChirpSize + j;
      float tolerance = 1e-5;
      std::complex<float> diff = toComp[i][j] - ref[globalIdx];
      if (abs(imag(diff)) > tolerance || abs(real(diff)) > tolerance) {
        printf("Miss match at rx[%d][%d] vs. reshaped[%d] \n", i, j, globalIdx);
        printf("input: %.3f %.3f expected %.3f %.3f \n", real(toComp[i][j]),
               imag(toComp[i][j]), real(ref[globalIdx]), imag(ref[globalIdx]));
        return false;
      }
    }
  }
  std::cout << "partiton Verification Passed\n" << std::endl;
  return true;
}

bool vectorVerification(std::vector<std::complex<float>> &input,
                        std::vector<std::complex<float>> &ref, int size) {
  // std::cout << "Vector Verification "<< std::endl;
  float tolerance = 1e-5;
  // doing fft for every ref input
  for (int i = 0; i < size; i++) {
    std::complex<float> diff = input[i] - ref[i];
    if (abs(imag(diff)) > tolerance || abs(real(diff)) > tolerance) {
      printf("Vector Verification failed at %d input: (%.3f %.3f) expected "
             "(%.3f %.3f)\n",
             i, real(input[i]), imag(input[i]), real(ref[i]), imag(ref[i]));
      return false;
    }
  }
  printf("Vector Verification passed\n");
  return true;
}

void showVector(std::vector<std::complex<float>> vec) {
  for (int i = 0; i < vec.size(); i++) {
    printf("vec[%d] real: %.3f  imag: %.3f \n", i, real(vec[i]), imag(vec[i]));
  }
}


class HostProducerClass;
class ReshapeClass;
class PartitionClass;
template <size_t id> class FFT1dWorker;
template <size_t id> class PartitionWorker;
class CopyBufferWorker;
class DistanceKernelClass;
class AngleKernelClass;
class KernelCompute;
class SpeedProducerKernel;
class SpeedReceiverKernel;
class SpeedTransposeKernel;
class SpeedFFTKernel1;
class SpeedFFTKernel2;
class SpeedFFTSwapKernel;

int main(int argc, char *argv[]) {
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
            << device.get_info<sycl::info::device::name>().c_str() << std::endl;
  // std::cout << "Reading raw data from binary file \n";
  unsigned result = 0;
  sycl::buffer<unsigned, 1> buffer_c(&result, 1);

  //  Values used as input to the kernel
  float kA = 2.0f;
  float kB = 3.0f;

  q.submit([&](sycl::handler &h) {
    // Accessor to the scalar result
    sycl::accessor accessor_c(buffer_c, h, sycl::write_only, sycl::no_init);

    // Kernel
    h.single_task<class KernelCompute>([=]() {
      // SyclSquare is a SYCL library function, defined in lib_sycl.cpp.
      float a_sq = SyclSquare(kA);
      float b_sq = SyclSquare(kB);

      // RtlByteswap is an RTL library.
      //  - When compiled for FPGA, Verilog module byteswap_uint in lib_rtl.v
      //    is instantiated in the datapath by the compiler.
      //  - When compiled for FPGA emulator (CPU), the C model of RtlByteSwap
      //    in lib_rtl_model.cpp is used instead.
      accessor_c[0] = RtlByteswap((unsigned)(a_sq + b_sq));
    });
  });
  q.wait();
  printf("result %u\n", result);

  char filepath[] = "../src/fhy_direct.bin";
  int data_per_frame = ChirpSize * SampleSize * numRx * 2;
  FILE *fp = fopen(filepath, "rb");
  if (fp == NULL) {
    std::cout << "cannot open file " << filepath << std::endl;
  }
  short *read_data = new short[data_per_frame];
  int size = 0;
  int frameCnt = 0;
  size = (int)fread(read_data, sizeof(short), data_per_frame, fp);
  printf("read %d elements\n", size);
  frameCnt++;
  /**
   * vector buffer declaration
   */
  // size_t extended_length = nextPow2(SampleSize * ChirpSize);
  std::vector<short> base_frame(read_data, read_data + data_per_frame);
  std::array<std::vector<std::complex<float>>, RxSize> partitioned_data;
  std::array<std::vector<std::complex<float>>, RxSize> fft_res;

  for (int i = 0; i < RxSize; i++) {
    partitioned_data[i].resize(extended_length);
    std::fill(partitioned_data[i].begin(), partitioned_data[i].end(), 0);
    fft_res[i].resize(extended_length);
  }
  std::vector<short> hold_consumer_internal(fifo_depth);
  std::vector<short> raw_input(data_per_frame);

  std::vector<std::complex<float>> reshaped_data(data_per_frame / 2);
  std::vector<std::complex<float>> phase_matrix(angleVecLenght * RxSize);
  std::vector<std::complex<float>> angle_weight(RxSize);
  std::vector<std::complex<float>> angle_vector(angleVecLenght, 0);
  std::vector<std::complex<float>> speed_rx0_extend(ChirpSize * ChirpSize, 0);
  std::vector<std::complex<float>> speed_transpose(ChirpSize * ChirpSize, 0);
  std::vector<std::complex<float>> speed_fft_res(ChirpSize * ChirpSize, 0);

  /**
   * phase_matrix initialization
   */
  for (int loc = 0; loc < RxSize; loc++) {
    for (int phi = -900; phi <= 900; phi++) {
      float theta = -loc * 2 * PI * d * sin((float)phi / 1800.0 * PI) / lamda;
      phase_matrix[loc * angleVecLenght + (phi + 900)] =
          std::complex(cos(theta), sin(theta));
      // printf("theta %f\n",theta);
    }
  }

  int pow = length2Pow(extended_length);
  while ((fread(read_data, sizeof(short), data_per_frame, fp)) > 0 && frameCnt < 100) {
    frameCnt++;
    printf("Reading frame %d \n", frameCnt);

    /**
     * preprocessing: reshape + partition
     */
    std::vector<short> tmpRead(read_data, read_data + data_per_frame);
    raw_input.swap(tmpRead);

    buffer base_frame_buffer(base_frame);
    buffer raw_input_buffer(raw_input);
    buffer reshaped_data_buffer(reshaped_data);
    // buffer partition_buffer(partitioned_data);
    auto hostProducerEvent = SubmitHostProducer<HostProducerClass, preProcessingPipe>(
            q, base_frame_buffer, raw_input_buffer);
    auto preProcWorkerEvent = SubmitPreProcWorker<ReshapeClass, PartitionClass, preProcessingPipe>(
            q, reshaped_data_buffer);

    /**
     * 1d fft data partition and angle detection
     */
    std::cout << "Enqueing FFT and Partition Worker " << std::endl;
    fpga_tools::UnrolledLoop<RxSize>([&](auto id) {
      buffer fftRes(fft_res[id]);
      buffer partitioned_buffer(partitioned_data[id]);
      auto fftWorkerEvent = SubmitParitionFFTWorker<PartitionWorker<id>, FFT1dWorker<id>, 
            id, extended_length, 
            extended_length>(q, partitioned_buffer, fftRes, pow);
    });
    q.wait();
    // assert(partitionVerification(partitioned_data, reshaped_data));

    /**
     * Speed Detection
     * 1. stage1 padding each chirp data in rx0 to be 2^n
     * 2. stage2 apply fft for each padded chirp of rx0 in chirp dimension
     * 3. stage3 transpose the transformed fft results
     * 4. stage4 apply fft in sample dimension and swap the front half and back half
     */
    buffer rx0_raw_buffer(partitioned_data[0]);
    buffer speed_rx0_extend_buffer(speed_rx0_extend);
    buffer speed_transpose_bufer(speed_transpose);
    buffer speed_fft_buffer(speed_fft_res);
    // std::vector<std::complex<float>> sample_transpose(ChirpSize * ChirpSize, 0);

    auto speedEvent = SubmitSpeedWorker<SpeedProducerKernel, SpeedReceiverKernel, SpeedTransposeKernel, 
                                        SpeedFFTKernel1,SpeedFFTKernel2,SpeedFFTSwapKernel,
                                        ChirpSize, ChirpSize * ChirpSize>(q,rx0_raw_buffer, speed_rx0_extend_buffer,
                                        speed_transpose_bufer, speed_fft_buffer);

    /**
     * Distance Detection: Result Verified
     */
    int maxIdx = findAbsMax(fft_res[0], floor(0.4 * extended_length));
    float maxDisIdx = maxIdx * ChirpSize * SampleSize / (float)extended_length;
    float Fs_extend = fs * extended_length / (ChirpSize * SampleSize);
    float maxDis = lightSpeed *
                   (((float)maxDisIdx / extended_length) * Fs_extend) /
                   (2 * mu);

    // printf("maxDisIdx %.3f maxDis %.3f  \n", maxDisIdx, maxDis);
    /**
     * Angle detection
     */
    // initialize angle weights
    for (int i = 0; i < RxSize; i++) {
      angle_weight[i] = fft_res[i][maxIdx];
      // printf("angle_weight[%d] %.3f %.3f\n",i,real(angle_weight[i]),
      // imag(angle_weight[i]));
    }

    buffer angle_weight_device(angle_weight);
    buffer phase_matrix_device(phase_matrix);
    buffer angle_vector_device(angle_vector);
    auto angleWorkerEvent = SubmitAngleWorker<AngleKernelClass>(q, angle_weight_device, phase_matrix_device, angle_vector_device);
    q.wait();
    /**
    * Angle Detection
    */
    int maxAngleIdx = findAbsMax(angle_vector, angleVecLenght);
    float angle = ((float)maxAngleIdx - 900.0) / 10.0;

    /**
     * Speed Detection
    */

    // for(int i = 0; i < ChirpSize * ChirpSize; i ++)
    // {
    //   printf("speed_fft_res[%d] real: %.3f img: %.3f \n",i, real(speed_fft_res[i]),imag(speed_fft_res[i]));
    // }
    int maxSpeedIdx = findAbsMax(speed_fft_res, ChirpSize * SampleSize) % ChirpSize;
    double fr = 1e6/64;
    double LAMDA = 3.0e08 / 77e9;
    double maxSpeed = ((double)maxSpeedIdx * fr/ChirpSize - fr/2) * LAMDA / 2;
    // printf("(double)maxSpeedIdx * fr/ChirpSize %.3f ((double)maxSpeedIdx * fr/ChirpSize - fr/2) %.3f\n",(double)maxSpeedIdx * fr/ChirpSize,((double)maxSpeedIdx * fr/ChirpSize - fr/2));
    printf("maxDis %.3f m, Angle %.3f degree, Speed %.3f m/s speed max index %d\n", maxDis, angle, maxSpeed, maxSpeedIdx);



  }

  fclose(fp);
  delete[] read_data;

  return 0;
}