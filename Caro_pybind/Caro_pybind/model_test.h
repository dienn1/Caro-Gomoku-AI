#include <pybind11/numpy.h>
#include "model.h"

namespace py = pybind11;

typedef Eigen::Array<float, 3, 3>  Array33f;
typedef Eigen::Vector<float, 256> Vector256f;
typedef Eigen::Array<float, 1, 1> Array11f;
typedef Eigen::Vector<float, 16> Vector16f;
typedef Eigen::Vector<float, 5> Vector5f;

typedef py::array_t<float> NumpyArray;


float SmallNetTest(Array77f(&example_input)[2], 
	const Array33f(&conv1Weights)[256 * 2], const Vector256f& conv1Bias, 
	const Array11f(&conv2Weights)[16 * 256], const Vector16f& conv2Bias,
	const MatrixXf& fc1Weights, const VectorXf& fc1Bias, 
	const MatrixXf& fc2Weights, const VectorXf& fc2Bias, 
	const MatrixXf& fc3Weights, const VectorXf& fc3Bias)
{
	SmallNet model(conv1Weights, conv1Bias, conv2Weights, conv2Bias, fc1Weights, fc1Bias, fc2Weights, fc2Bias, fc3Weights, fc3Bias);
	return model.forward(example_input);
}


float py_SmallNetTest(NumpyArray example_input,
	const NumpyArray conv1Weights, const NumpyArray conv1Bias,
	const NumpyArray conv2Weights, const NumpyArray conv2Bias,
	const NumpyArray fc1Weights, const NumpyArray fc1Bias,
	const NumpyArray fc2Weights, const NumpyArray fc2Bias,
	const NumpyArray fc3Weights, const NumpyArray fc3Bias)
{
	Array77f input_eigen[2];

	Array33f conv1Weights_eigen[256 * 2];
	Vector256f conv1Bias_eigen;

	Array11f conv2Weights_eigen[16 * 256];
	Vector16f conv2Bias_eigen;

	MatrixXf fc1Weights_eigen(800, 400);
	VectorXf fc1Bias_eigen(800);

	MatrixXf fc2Weights_eigen(256, 800);
	VectorXf fc2Bias_eigen(256);

	MatrixXf fc3Weights_eigen(1, 256);
	VectorXf fc3Bias_eigen(1);

	// convert input
	auto t_input = example_input.unchecked<3>();
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			for (int k = 0; k < 7; k++)
			{
				input_eigen[i](j, k) = t_input(i, j, k);
			}
		}
	}

	// convert conv1Weights
	auto t_conv1Weights = conv1Weights.unchecked<4>();
	for (int out_channel = 0; out_channel < 256; out_channel++)
	{
		for (int in_channel = 0; in_channel < 2; in_channel++)
		{
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					conv1Weights_eigen[out_channel * 2 + in_channel](i, j) = t_conv1Weights(out_channel, in_channel, i, j);
				}
			}
		}
	}
	// convert conv1Bias
	auto t_conv1Bias = conv1Bias.unchecked<1>();
	for (int out_channel = 0; out_channel < 256; out_channel++)
	{
		conv1Bias_eigen(out_channel) = t_conv1Bias(out_channel);
	}

	// convert conv1Weights
	auto t_conv2Weights = conv2Weights.unchecked<4>();
	for (int out_channel = 0; out_channel < 16; out_channel++)
	{
		for (int in_channel = 0; in_channel < 256; in_channel++)
		{
			conv2Weights_eigen[out_channel * 256 + in_channel](0, 0) = t_conv2Weights(out_channel, in_channel, 0, 0);
		}
	}
	// convert conv1Bias
	auto t_conv2Bias = conv2Bias.unchecked<1>();
	for (int out_channel = 0; out_channel < 16; out_channel++)
	{
		conv2Bias_eigen(out_channel) = t_conv2Bias(out_channel);
	}

	// convert fc1Weights, fc1Bias
	auto t_fc1Weights = fc1Weights.unchecked<2>();
	auto t_fc1Bias = fc1Bias.unchecked<1>();
	for (int i = 0; i < 800; i++)
	{
		fc1Bias_eigen(i) = t_fc1Bias(i);
		for (int j = 0; j < 400; j++)
		{
			fc1Weights_eigen(i, j) = t_fc1Weights(i, j);
		}
	}

	// convert fc2Weights, fc2Bias
	auto t_fc2Weights = fc2Weights.unchecked<2>();
	auto t_fc2Bias = fc2Bias.unchecked<1>();
	for (int i = 0; i < 256; i++)
	{
		fc2Bias_eigen(i) = t_fc2Bias(i);
		for (int j = 0; j < 800; j++)
		{
			fc2Weights_eigen(i, j) = t_fc2Weights(i, j);
		}
	}

	// convert fc2Weights, fc2Bias
	auto t_fc3Weights = fc3Weights.unchecked<2>();
	auto t_fc3Bias = fc3Bias.unchecked<1>();
	fc3Bias_eigen(0) = t_fc3Bias(0);
	for (int j = 0; j < 256; j++)
	{
		fc3Weights_eigen(0, j) = t_fc3Weights(0, j);
	}

	float out = SmallNetTest(input_eigen,
		conv1Weights_eigen, conv1Bias_eigen,
		conv2Weights_eigen, conv2Bias_eigen,
		fc1Weights_eigen, fc1Bias_eigen,
		fc2Weights_eigen, fc2Bias_eigen,
		fc3Weights_eigen, fc3Bias_eigen);

	return out;
}