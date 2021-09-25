
#include <IntIntSoftmax.h>
#include <QuantAct.h>

IntIntSoftmax::IntIntSoftmax(
		int Output_bit,
		string Quant_mode = "none",
		string Force_dequant = "none")
	{
		output_bit = Output_bit;
		quant_mode = Quant_mode;
		if (Force_dequant == "nonlinear")
		{
			cout << "Force dequantize softmax" << endl;
			quant_mode = 'none';
		}
		else if (Force_dequant == "softmax")
		{
			cout << "Force dequantize softmax" << endl;
			quant_mode = 'none';
		}
		coef[1] /= coef[0];
		coef[2] /= coef[0];
	}

	torch::Tensor IntIntSoftmax::int_polynomial(torch::Tensor x_int, torch::Tensor& scaling_factor)
	{
		torch::Tensor b_int, c_int;
		torch::Tensor z;
		{
			torch::NoGradGuard no_grad;
			b_int = floor(coef[1] / (scaling_factor));
			c_int = floor(coef[2] / ((scaling_factor) * (scaling_factor)));
		}

		z = x_int + b_int;
		z = x_int * z;
		z = z + c_int;
		scaling_factor = coef[0] * (scaling_factor) * (scaling_factor);

		return z;
	}

	torch::Tensor IntIntSoftmax::int_exp(torch::Tensor& x_int, torch::Tensor& scaling_factor)
	{
		torch::Tensor x0_int;
		{
			torch::NoGradGuard no_grad;
			x0_int = torch::floor(x0 / scaling_factor);
		}
		x_int = torch::max(x_int, n * x0_int);
		torch::Tensor q = torch::floor(x_int / x0_int);
		torch::Tensor r = x_int - x0_int * q;
		torch::Tensor exp_scalingfactor = scaling_factor;
		torch::Tensor exp_int = int_polynomial(r, exp_scalingfactor);
		exp_int = torch::clamp(torch::floor(exp_int * (2 ^ (n - q))), 0);
		scaling_factor = exp_scalingfactor / (2 ^ n);
		return exp_int;
	}

	torch::Tensor IntIntSoftmax::forward(torch::Tensor x, torch::Tensor scaling_factor)
	{
		/*if (quant_mode == "none")
		{
			return softmax(x, dim = -1, onnx_trace = false);
		}*/
		if (quant_mode != "symmetric")
		{
			printf("unspported mode!");
			exit(-1);
		}

		torch::Tensor x_int;

		x_int = x / scaling_factor;
		std::tuple<torch::Tensor, torch::Tensor> max_tempo = torch::max(x_int, -1, true);
		auto x_int_max = std::get<0>(max_tempo);
		x_int = x_int - x_int_max;

		torch::Tensor exp_int, exp_int_sum, exp;
		torch::Tensor exp_scaling_factor = scaling_factor;

		exp_int = int_exp(x_int, exp_scaling_factor);
		QuantAct QAct(16, "symmetric");
		exp = QAct.forward(exp_int, exp_scaling_factor);
		exp_int = exp / exp_scaling_factor;
		exp_int_sum = torch::sum(exp_int, -1, true);

		auto factor = floor((2 ^ 32) / exp_int_sum);
		exp_int = floor(exp_int * factor / (2 ^ (32 - output_bit)));
		scaling_factor = torch::tensor(1 / ((2) ^ output_bit));
		return exp_int * scaling_factor;
	}