#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../../include/exception_handler.hpp"
#include <complex>
#include "kernel.hpp"
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

int length2Pow(int len)
{
    int cnt = 0;
    int tmp = 1;
    while(tmp < len)
    {
        tmp <= 1;
        cnt ++;
    }
    return cnt;
}


bool partitionVerification(std::array<std::vector<std::complex<double>>, RxSize> toComp, std::vector<std::complex<double>> ref)
{
    for (int i = 0; i < RxSize; i++)
    {
        // printf("Rx[%d] size %d \n", i, partitioned_data[i].size());
        for (int j = 0; j < SampleSize * ChirpSize; j++)
        {
            // printf("data[%d] (%.3f %.3f) \n", j, real(toComp[i][j]), imag(toComp[i][j]));
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
        printf("fftRes[%d] (%.3f  %.3f) ref[%d] (%.3f %.3f)\n", i, real(fftRes[i]), imag(fftRes[i]), i, real(ref[i]), imag(ref[i]));
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

void showVector(std::vector<std::complex<double>> vec)
{
    for(size_t i = 0; i < vec.size(); i ++)
    {
        printf("vec[%d] real: %.3f  imag: %.3f \n",i, real(vec[i]), imag(vec[i]));
    }
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
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL)
    {
        std::cout << "cannot open file " << filepath << std::endl;
    }
    short *read_data = new short[data_per_frame];
    int size = 0;
    int frameCnt = 0;
    size = (int)fread(read_data, sizeof(short), data_per_frame, fp);
    printf("read %d elements\n",size);
    frameCnt++;
    /**
     * vector buffer declaration
     */
    // size_t extended_length = nextPow2(SampleSize * ChirpSize);
    std::vector<short> base_frame(read_data, read_data + data_per_frame);
    std::array<std::vector<std::complex<double>>, RxSize> partitioned_data;
    std::array<std::vector<std::complex<double>>, RxSize> fft_res;
    for(int i = 0; i < RxSize; i ++)
    {
        partitioned_data[i].resize(extended_length - SampleSize * ChirpSize);
        std::fill(partitioned_data[i].begin(),partitioned_data[i].end(),0);
        fft_res[i].resize(extended_length);
    }
    std::vector<short> hold_consumer_internal(fifo_depth);
    std::vector<short> raw_input(data_per_frame);

    std::vector<std::complex<double>> reshaped_data(data_per_frame / 2);
    std::vector<std::complex<double>> preproc_buffer(data_per_frame / 2);

    int pow = length2Pow(extended_length);

    if ((fread(read_data, sizeof(short), data_per_frame, fp)) > 0 && frameCnt < 100)
    {
        frameCnt++;
        printf("Reading frame %d \n", frameCnt);
        printf("read %d elements\n",size);

        /**
         * preprocessing: reshape + partition
        */
        std::vector<short>tmpRead(read_data, read_data+data_per_frame);
        raw_input.swap(tmpRead);

        buffer base_frame_buffer_device(base_frame);
        buffer producer_buffer_device(raw_input);
        size_t num_elements = raw_input.size();
        buffer consumer_buffer_device(reshaped_data);
        buffer preproc_buffer_device(preproc_buffer);
        // buffer partition_buffer(partitioned_data);
        auto preprocProducerEvent = PreProcessingProducer(q, base_frame_buffer_device, producer_buffer_device);
        auto preprocConsumerEvent = PreProcessingConsumer(q, consumer_buffer_device);
        q.wait();
        auto partition_event = q.submit([&](handler &h){
            fpga_tools::UnrolledLoop<RxSize>([&](auto i){
                auto it = partitioned_data[i].begin();
                int start = i * SampleSize * ChirpSize;
                int end = (i + 1) * SampleSize * ChirpSize;
                partitioned_data[i].insert(it, reshaped_data.begin() + start, reshaped_data.begin() + end);
            });
        });
        partition_event.wait();
        assert(partitionVerification(partitioned_data, reshaped_data));

        /**
         * FFT in the device
        */
        q.wait();
        fpga_tools::UnrolledLoop<RxSize>([&](auto id){
            buffer fftInput(partitioned_data[id]);
            buffer fftOutput(fft_res[id]);
            // std::cout << fftInput.size() << " " << fftOutput.size() << std::endl;
            auto fftProducerEvent = fftProducer<id>(q, fftInput);
            auto fftConsumerEvent = fftConsumer<id,extended_length, extended_length>(q, fftOutput, pow);
        });
        q.wait();
        
        // showVector(fft_res[0]);

        std::cout << frameCnt << std::endl;
        for(int i = 0; i < RxSize; i ++)
        {
            partitioned_data[i].resize(extended_length - SampleSize * ChirpSize);
            std::fill(partitioned_data[i].begin(),partitioned_data[i].end(),0);
            fft_res[i].resize(extended_length);
        }
    }

    fclose(fp);
    delete[] read_data;
    return 0;
}