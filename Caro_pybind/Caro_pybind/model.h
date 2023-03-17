#pragma once

#include <iostream>
#include <Eigen>
#include <ctime>
#include <math.h>


using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXXf;
using Eigen::Array33f;
typedef Eigen::Array<float, 7, 7> Array77f;


MatrixXf Relu(const MatrixXf& x)
{
	MatrixXf A(x.rows(), x.cols());
	for (int i = 0; i < x.size(); ++i)
	{
		if (x(i) < 0) 
		{
			A(i) = 0;
		}
		else
		{
			A(i) = x(i);
		}
	}
	return std::move(A);
}



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

	float forward(const Array77f(&x)[2])
	{
		auto x1 = conv1(x);
		conv1.Relu(x1);
		auto x2 = conv2(x1);
		conv2.Relu(x2);

		VectorXf flatten_x2(400);
		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				for (int k = 0; k < 5; k++)
				{
					flatten_x2(i * 5 * 5 + j * 5 + k) = x2[i](j, k);
				}
			}
		}

		auto x3 = Relu(fc1(flatten_x2));
		auto x4 = Relu(fc2(x3));
		auto x5 = fc3(x4);

		delete[] x1;
		delete[] x2;

		return tanh(x5(0));
	}

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

	float forward(const Array77f(&x)[2])
	{
		auto x1 = conv1(x);
		conv1.Relu(x1);

		VectorXf flatten_x1(125);
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				for (int k = 0; k < 5; k++)
				{
					flatten_x1(i * 5 * 5 + j * 5 + k) = x1[i](j, k);
				}
			}
		}

		delete[] x1;

		auto x2 = Relu(fc1(flatten_x1));
		auto x3 = fc2(x2);

		return tanh(x3(0));
	}

private:
	Convolve<2, 5, 3, 7> conv1;
	Linear<125, 50> fc1;
	Linear<50, 1> fc2;
};
