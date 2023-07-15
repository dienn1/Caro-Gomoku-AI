#include "model.h"

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

float TestNet::forward(const Array77f(&x)[2])
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

float SmallNet::forward(const Array77f(&x)[2])
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

float SmallNet::py_forward(NumpyArray py_input)
{
	Array77f input_eigen[2];

	// convert input
	auto t_input = py_input.unchecked<3>();
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

	return forward(input_eigen);
}

SmallNet SmallNet::py_Create(const NumpyArray conv1Weights, const NumpyArray conv1Bias,
	const NumpyArray conv2Weights, const NumpyArray conv2Bias,
	const NumpyArray fc1Weights, const NumpyArray fc1Bias,
	const NumpyArray fc2Weights, const NumpyArray fc2Bias,
	const NumpyArray fc3Weights, const NumpyArray fc3Bias)
{
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

	return SmallNet(conv1Weights_eigen, conv1Bias_eigen,
		conv2Weights_eigen, conv2Bias_eigen,
		fc1Weights_eigen, fc1Bias_eigen,
		fc2Weights_eigen, fc2Bias_eigen,
		fc3Weights_eigen, fc3Bias_eigen);
}