#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#ifdef OPENCL_RUNTIME
#include <libclew/ocl_init.h>
#endif

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

std::unique_ptr<double[]> read_matrix(size_t matrix_size, std::ifstream& is)
{
	std::unique_ptr<double[]> result(new double[matrix_size]());
	for (size_t i = 0; i < matrix_size; i++)
	{
		is >> result.get()[i];
	}
	return result;
}


int main()
{
#ifdef OPENCL_RUNTIME
	if (!ocl_init())
	{
		std::cerr << ("Can't init OpenCL driver!") << "\n";
		return 1;
	}
#endif
#ifdef DEBUG
	std::ofstream input_fill("input.txt");
	int N = 1023;
	int M = 9;

	input_fill << N << " " << M << "\n";
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			input_fill << 1 << " ";
		}
		input_fill << "\n";
	}
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			input_fill << 1 << " ";
		}
		input_fill << "\n";
	}
	input_fill.flush();
#endif
	std::ifstream input("input.txt");
	std::ofstream output("output.txt");
	int n, m;
	input >> n >> m;
	size_t first_matrix_size = n * n;
	size_t second_matrix_size = m * m;
	auto first_matrix = read_matrix(first_matrix_size, input);
	auto second_matrix = read_matrix(second_matrix_size, input);
	std::unique_ptr<double[]> result(new double[first_matrix_size]());

	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	std::vector<cl::Kernel> kernels;
	try
	{
		cl::Platform::get(&platforms);
#ifdef DEBUG
		platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
#else 
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
#endif
		cl::Context context(devices);
		cl::CommandQueue queue(context, devices[0]);
		std::ifstream cl_file("matrix_convolution.cl");
		std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
		                                              cl_string.length() + 1));
		cl::Program program(context, source);
		const size_t block_size = 16;
		const auto block_size_option = "-D BLOCK_SIZE=" + std::to_string(block_size);
		program.build(devices, block_size_option.c_str());
		cl::Buffer dev_first_matrix(context, CL_MEM_READ_ONLY, sizeof(double) * first_matrix_size);
		cl::Buffer dev_second_matrix(context, CL_MEM_READ_ONLY, sizeof(double) * second_matrix_size);
		cl::Buffer dev_result(context, CL_MEM_WRITE_ONLY, sizeof(double) * first_matrix_size);
		queue.enqueueWriteBuffer(dev_first_matrix, CL_TRUE, 0, sizeof(double) * first_matrix_size, first_matrix.get());
		queue.enqueueWriteBuffer(dev_second_matrix, CL_TRUE, 0, sizeof(double) * second_matrix_size, second_matrix.get());
		cl::Kernel kernel(program, "matrix_convolution");
		const int next_dividable = (n + block_size - 1) / block_size * block_size;
		cl::KernelFunctor matrix_convolution(kernel, queue, cl::NullRange, cl::NDRange(next_dividable, next_dividable),
		                                     cl::NDRange(block_size, block_size));
		matrix_convolution(dev_first_matrix, dev_second_matrix, dev_result, n, m);
		queue.enqueueReadBuffer(dev_result, CL_TRUE, 0, sizeof(double) * first_matrix_size, result.get());
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				output << result.get()[i * n + j] << " ";
			}
			output << "\n";
		}
		output.flush();
	}
	catch (cl::Error& e)
	{
		std::cerr << std::endl << e.what() << " : " << e.err() << "\n";
	}

	return 0;
}
