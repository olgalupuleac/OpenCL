#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <memory>

#include "cl.hpp"

#define DEBUG
const size_t block_size = 256;

const auto number_of_bytes(size_t n)
{
    return n * sizeof(double);
}

const auto next_dividable_number(size_t n)
{
    return (n + block_size - 1) / block_size * block_size;
}

std::vector<double> final_sum(std::vector<double> &array, std::vector<double> sums,
        cl::Program &program, cl::Context &context, cl::CommandQueue &queue)
{
    std::vector<double> result(array.size());
    size_t array_size = number_of_bytes(array.size());
    size_t sums_size = number_of_bytes(sums.size());
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, array_size);
    cl::Buffer dev_sum_input(context, CL_MEM_READ_ONLY, array_size);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, array_size);
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, array_size, array.data());
    queue.enqueueWriteBuffer(dev_sum_input, CL_TRUE, 0, sums_size, sums.data());
    cl::Kernel kernel(program, "final_sum");
    cl::KernelFunctor final_sum_kernel(kernel, queue, cl::NullRange, cl::NDRange(next_dividable_number(array.size())),
                                       cl::NDRange(block_size));
    final_sum_kernel(dev_input, dev_sum_input, dev_output, (int)array.size());
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, array_size, result.data());
    return result;
}

std::vector<double> prefix_sum(std::vector<double> &array, cl::Program &program,
                               cl::Context &context, cl::CommandQueue &queue) {
    size_t n = array.size();

    std::vector<double> result(n);
    const size_t next_dividable = next_dividable_number(n);
    const size_t data_size = number_of_bytes(n);
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, data_size);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, data_size);
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, data_size, array.data());

    cl::Kernel kernel(program, "prefix_sum");

    cl::KernelFunctor prefix_sum_kernel(kernel, queue, cl::NullRange,
                                        cl::NDRange(next_dividable),
                                        cl::NDRange(block_size));
    prefix_sum_kernel(dev_input, dev_output,
                      cl::__local(block_size * sizeof(double)),
                      cl::__local(block_size * sizeof(double)), (int) n);

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, data_size, result.data());

    if (n < block_size) {
        return result;
    }
    size_t num_of_blocks = next_dividable / block_size;
    std::vector<double> block_sums(num_of_blocks);
    for (int i = 0; i < num_of_blocks; i++) {
        block_sums[i] = result[block_size * i - 1];
    }
    std::vector<double> recursive_call_sums = prefix_sum(block_sums, program,
                                                         context, queue);
    return final_sum(result, recursive_call_sums, program, context, queue);
}

int main() {
#ifdef DEBUG
    std::ofstream input_fill("input.txt");
    int N = 1024;

    input_fill << N << "\n";
    for (int i = 0; i < N; i++)
    {
        input_fill << 1 << ' ';
    }
    input_fill << std::endl;
#endif
    std::ifstream input("input.txt");
    std::ofstream output("output.txt");
    size_t n;
    input >> n;
    std::vector<double> initial_array(n);
    for (size_t i = 0; i < n; i++) {
        input >> initial_array[i];
    }

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;
    try
    {
        cl::Platform::get(&platforms);

        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0]);
        std::ifstream cl_file("prefix_sum.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file),
                              (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));
        cl::Program program(context, source);

        const std::string block_size_option =
                "-D BLOCK_SIZE=" + std::to_string(block_size);
        program.build(devices, block_size_option.c_str());

        auto result = prefix_sum(initial_array, program, context, queue);


        for (int i = 0; i < n; i++)
        {
            output << result[i] << " ";
        }
        output << std::endl;
    }
    catch (cl::Error &e)
    {
        std::cerr << std::endl << e.what() << " : " << e.err() << "\n";
    }

    return 0;
}