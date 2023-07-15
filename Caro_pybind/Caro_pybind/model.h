#pragma once

#include <iostream>
#include <Eigen>
#include <ctime>
#include <math.h>
#include <pybind11/numpy.h>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXXf;
using Eigen::Array33f;
typedef Eigen::Array<float, 7, 7> Array77f;

namespace py = pybind11;
typedef py::array_t<float> NumpyArray;


template <unsigned int in_features, unsigned int out_features>
class Linear
{
public:
	Linear(const MatrixXf& _weights, const MatrixXf& _bias):
		weights(_weights), bias(_bias)
	{
		if (_weights.rows() != out_features || _weights.cols() != in_features)
		{
			throw std::invalid_argument("Weights dimensions does not match with in_features and out_features.");
		}
		if (_bias.rows() != out_features)
		{
			throw std::invalid_argument("Bias dimension does not match with out_features.");
		}
	}

	VectorXf operator()(VectorXf x) const
	{
		return x.transpose() * weights.transpose() + bias.transpose();
	}

private:
	MatrixXf weights;
	VectorXf bias;
};


template <unsigned int in_channels, unsigned int out_channels, unsigned int kernel_size, unsigned int input_size>
class Convolve
{
	typedef Eigen::Array<float, kernel_size, kernel_size> ArrayKernel;
	typedef Eigen::Vector<float, out_channels> VectorBias;

	typedef Eigen::Array<float, input_size, input_size> ArrayInput;

	const static unsigned int out_dim = input_size - kernel_size + 1;
	typedef Eigen::Array<float, out_dim, out_dim> ArrayOutput;


public:
	Convolve(const ArrayKernel (& _weights)[out_channels * in_channels], const VectorBias& _bias)
	{
		// weights = _weights;
		for (int i = 0; i < out_channels * in_channels; i++)
		{
			weights[i] = _weights[i];
		}
		bias = _bias;
	}

	ArrayOutput* operator()(const ArrayInput* x)
	{
		ArrayOutput* out = new ArrayOutput[out_channels];
		
		for (int i = 0; i < out_channels; i++)	// For every out channel
		{
			ArrayOutput out_channel;
			for (int x1 = 0; x1 < out_dim; x1++)
			{
				for (int x2 = 0; x2 < out_dim; x2++)
				{
					// Convolve
					float element = 0;
					
					for (int j = 0; j < in_channels; j++)	// For every in channel
					{
						element += (weights[(i * in_channels) + j] * x[j].block(x1, x2, kernel_size, kernel_size)).sum();
					}

					out_channel(x1, x2) = element + bias[i];
				}
			}
			out[i] = out_channel;
		}
		return out;
	}

	// in_place Relu
	static void Relu(ArrayOutput* x)
	{
		for (int i = 0; i < out_channels; i++)	// For every out channel
		{
			for (int j = 0; j < out_dim * out_dim; j++)
			{
				if (x[i](j) < 0)
				{
					x[i](j) = 0;
				}
			}
		}
	}

private:
	ArrayKernel weights[out_channels * in_channels]; // should be accessed as weights[out_channel*IN_CHANNELS + in_channel] ( = weights[out_channel][in_channel] )
	VectorBias bias;
};



class SmallNet
{
public:
	typedef Eigen::Array<float, 3, 3>  Array33f;
	typedef Eigen::Vector<float, 256> Vector256f;
	
	typedef Eigen::Array<float, 1, 1> Array11f;
	typedef Eigen::Vector<float, 16> Vector16f;


	SmallNet(const Array33f (&conv1Weights)[256*2], const Vector256f& conv1Bias, 
		const Array11f (&conv2Weights)[16*256], const Vector16f& conv2Bias,
		const MatrixXf& fc1Weights, const VectorXf& fc1Bias, 
		const MatrixXf& fc2Weights, const VectorXf& fc2Bias, 
		const MatrixXf& fc3Weights, const VectorXf& fc3Bias):
		conv1(conv1Weights, conv1Bias), conv2(conv2Weights, conv2Bias), fc1(fc1Weights, fc1Bias), fc2(fc2Weights, fc2Bias), fc3(fc3Weights, fc3Bias)
	{

	}

	// Python-binding factory function
	static SmallNet py_Create(const NumpyArray conv1Weights, const NumpyArray conv1Bias,
		const NumpyArray conv2Weights, const NumpyArray conv2Bias,
		const NumpyArray fc1Weights, const NumpyArray fc1Bias,
		const NumpyArray fc2Weights, const NumpyArray fc2Bias,
		const NumpyArray fc3Weights, const NumpyArray fc3Bias);

	float forward(const Array77f(&x)[2]);

	float py_forward(NumpyArray py_input);

private:
	Convolve<2, 256, 3, 7> conv1;
	Convolve<256, 16, 1, 5> conv2;
	Linear<400, 800> fc1;
	Linear<800, 256> fc2;
	Linear<256, 1> fc3;
};


class TestNet
{
public:
	typedef Eigen::Array<float, 3, 3>  Array33f;
	typedef Eigen::Vector<float, 5> Vector5f;


	TestNet(const Array33f(&conv1Weights)[5 * 2], const Vector5f& conv1Bias, 
		const MatrixXf& fc1Weights, const VectorXf& fc1Bias, const MatrixXf& fc2Weights, const VectorXf& fc2Bias) :
		conv1(conv1Weights, conv1Bias), fc1(fc1Weights, fc1Bias), fc2(fc2Weights, fc2Bias)
	{

	}

	float forward(const Array77f(&x)[2]);

private:
	Convolve<2, 5, 3, 7> conv1;
	Linear<125, 50> fc1;
	Linear<50, 1> fc2;
};
