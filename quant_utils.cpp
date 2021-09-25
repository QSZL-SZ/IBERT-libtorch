#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <quant_utils.h>

using namespace std;

namespace F = torch::nn::functional;

int batch_frexp(torch::Tensor& inputs, torch::Tensor& output_m, torch::Tensor& output_e, int max_bit = 31)
{
	auto shape_of_input = inputs.sizes();

	inputs = inputs.view({ -1 });

	//numpy based function need to be implemented

	return 1;
}

torch::Tensor symmetric_linear_quantization_params(int num_bits, torch::Tensor &saturation_min, torch::Tensor &saturation_max, bool per_channel = false)
{
	int n;
	{
		torch::NoGradGuard no_grad;
		n = 2 ^ (num_bits - 1) - 1;
	}
	if (per_channel)
	{
		std::tuple<torch::Tensor, torch::Tensor> maxval = torch::max(torch::stack({ saturation_min.abs(), saturation_max.abs() }, 1), 1);
		auto scale = std::get<0>(maxval);
		scale = scale.clamp(1e-8, INT_MAX * 1.0) / n;
		return scale;
	}
	else
	{
		auto scale = max(saturation_min.abs(), saturation_max.abs());
		scale = scale.clamp(1e-8, INT_MAX * 1.0) / n;
		return scale;
	}
}

torch::Tensor linear_quantize(torch::Tensor &input, torch::Tensor &scale, torch::Tensor &zero_point, bool inplace = false)
{
	if (sizeof(input.sizes()) == 4)
	{
		scale = scale.reshape({ -1, 1, 1, 1 });
		zero_point = zero_point.reshape({ -1, 1, 1, 1 });
	}
	else if (sizeof(input.sizes()) == 2)
	{
		scale = scale.reshape({ -1, 1 });
		zero_point = zero_point.reshape({ -1, 1 });
	}
	else
	{
		scale = scale.reshape({ -1 });
		zero_point = zero_point.reshape({ -1 });
	}

	if (inplace)
	{
		input.mul_(1. / scale).add_(zero_point).round_();
		return input;
	}

	return torch::round(1. / scale * input + zero_point);
}