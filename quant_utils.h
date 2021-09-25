#pragma once
#ifndef _QUANT_UTILS_H_
#define _QUANT_UTILS_H_

#include <torch/torch.h>
#include <assert.h>
namespace F = torch::nn::functional;
using namespace std;

torch::Tensor symmetric_linear_quantization_params(int num_bits, torch::Tensor &saturation_min, torch::Tensor &saturation_max, bool per_channel = false);

torch::Tensor linear_quantize(torch::Tensor &input, torch::Tensor &scale, torch::Tensor &zero_point, bool inplace = false);

int batch_frexp(torch::Tensor& inputs, torch::Tensor& output_m, torch::Tensor& output_e, int max_bit);

class SymmetricQuantFunction : torch::nn::Functional {

public:
	torch::Tensor forward(torch::Tensor x, int k, bool percentile_mode = false, torch::Tensor specified_scale = torch::empty({ 0 }))
	{
		if (specified_scale.numel() != 0)
		{
			scale = specified_scale;
		}

		torch::Tensor zeropoint = torch::zeros({ 1 });

		int n = 2 ^ (k - 1) - 1;
		torch::Tensor new_quant_x = linear_quantize(x, scale, zeropoint, false);
		new_quant_x = torch::clamp(new_quant_x, -n, n - 1);

		return new_quant_x;
	}
private:

	torch::Tensor scale = torch::tensor({ 1 });
};

class fixedpoint_mul : torch::nn::Functional {

public:
	torch::Tensor forward(torch::Tensor pre_act,
		torch::Tensor pre_act_scaling_factor,
		int bit_num,
		string quant_mode,
		torch::Tensor Z_scaling_factor,
		torch::Tensor Identity = torch::empty({ 0 }),
		torch::Tensor identity_scaling_factor = torch::empty({ 0 }))
	{
		identity = Identity;
		int n;
		if (quant_mode == "symmetric")
		{
			n = 2 ^ (bit_num - 1) - 1;
		}
		else
			n = 2 ^ (bit_num - 1);

		{
			torch::NoGradGuard no_grad;
			if (pre_act_scaling_factor.sizes().size() != 3)
			{
				pre_act_scaling_factor = pre_act_scaling_factor.reshape({ 1,1,-1 });
			}
			if (identity.numel() != 0)
			{
				if (pre_act_scaling_factor.sizes().size() != 3)
				{
					identity_scaling_factor = identity_scaling_factor.reshape({ 1,1,-1 });
				}
			}

			z_scaling_factor = Z_scaling_factor;
			auto z_int = torch::round(pre_act / pre_act_scaling_factor);
			auto _A = pre_act_scaling_factor.to(torch::kDouble);
			auto _B = (z_scaling_factor.to(torch::kFloat)).to(torch::kDouble);
			auto new_scale = _A / _B;
			if (pre_act_scaling_factor.sizes().size() != 3)
			{
				new_scale = new_scale.reshape({ 1,1,-1 });
			}

			torch::Tensor m, e;
			batch_frexp(new_scale, m, e); // need to implement

			torch::Tensor output = z_int.to(torch::kDouble) * m.to(torch::kDouble);
			output = torch::round(output / (2.0 ^ (e)));

			if (identity.numel() != 0)
			{
				auto _A = identity_scaling_factor.to(torch::kDouble);
				auto _B = (z_scaling_factor.to(torch::kFloat)).to(torch::kDouble);
				new_scale = _A / _B;
				if (pre_act_scaling_factor.sizes().size() != 3)
				{
					new_scale = new_scale.reshape({ 1,1,-1 });
				}
				torch::Tensor m1, e1;
				batch_frexp(new_scale, m1, e1);

				torch::Tensor output1 = z_int.to(torch::kDouble) * m.to(torch::kDouble);
				output1 = torch::round(output / (2.0 ^ (e)));

				output = output1 + output;
			}

			if (bit_num == 4 || bit_num == 8 || bit_num == 16)
			{
				if (quant_mode == "symmetric")
				{
					return torch::clamp(output.to(torch::kFloat), -n - 1, n);
				}
				else
				{
					return torch::clamp(output.to(torch::kFloat), 0, n);
				}
			}
			else
			{
				return output.to(torch::kFloat);
			}
		}


		return torch::empty({ 0 });
	}
private:
	torch::Tensor identity;
	torch::Tensor z_scaling_factor;
	torch::Tensor scale = torch::tensor({ 1 });
};


#endif
