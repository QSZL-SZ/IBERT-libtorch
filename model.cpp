#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <assert.h>

using namespace std;

namespace F = torch::nn::functional;


#include <torch/torch.h>

torch::Tensor symmetric_linear_quantization_params(int num_bits, torch::Tensor saturation_min, torch::Tensor saturation_max, bool per_channel = false)
{
	int n;
	{
		torch::NoGradGuard no_grad;
		n = 2 ^ (num_bits - 1) - 1;
	}
	if (per_channel)
	{
		std::tuple<torch::Tensor, torch::Tensor> maxval = torch::max(torch::stack({ saturation_min.abs(), saturation_max.abs() },1), 1);
		auto scale = std::get<0>(maxval);
		scale = scale.clamp(1e-8, INT_MAX * 1.0)/n;
		return scale;
	}
	else
	{
		auto scale = max(saturation_min.abs(), saturation_max.abs());
		scale = scale.clamp(1e-8, INT_MAX * 1.0) / n;
		return scale;
	}
}

torch::Tensor linear_quantize(torch::Tensor input, torch::Tensor scale, torch::Tensor zero_point, bool inplace = false)
{
	if (sizeof(input.sizes()) == 4)
	{
		scale = scale.reshape({ -1, 1, 1, 1 });
		zero_point = zero_point.reshape({ -1, 1, 1, 1 });
	}
	else if (sizeof(input.sizes()) == 2)
	{
		scale = scale.reshape({ -1, 1});
		zero_point = zero_point.reshape({ -1, 1});
	}
	else
	{
		scale = scale.reshape({ -1});
		zero_point = zero_point.reshape({ -1});
	}

	if (inplace)
	{
		input.mul_(1. / scale).add_(zero_point).round_();
		return input;
	}

	return torch::round(1. / scale * input + zero_point);
}

struct SymmetricQuantFunction : torch::nn::Functional {
	

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

	torch::Tensor scale = torch::tensor({ 1 });
};

struct fixedpoint_mul : torch::nn::Functional {


	torch::Tensor forward(torch::Tensor pre_act,
		torch::Tensor pre_act_scaling_factor,
		int bit_num,
		string quant_mode,
		torch::Tensor Z_scaling_factor,
		torch::Tensor Identity = torch::empty({ 0 }),
		torch::Tensor identity_scaling_factor = torch::empty({ 0 }))
	{
		/*if (pre_act_scaling_factor.sizes().size() == 3)
			auto xreshape = [](torch::Tensor x) {return x; };
		else
			auto xreshape = [](torch::Tensor x) {return x.reshape({1,1,-1}); };*/
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

	torch::Tensor identity;
	torch::Tensor z_scaling_factor;
	torch::Tensor scale = torch::tensor({ 1 });
};



struct QuantAct : torch::nn::Module {
	QuantAct(int Activation_bit,
		string Quant_mode = "none",
		double Act_range_momentum = 0.95,
		bool Running_stat = true,
		bool Per_channel = false,
		bool channel_len = NULL) {

		activation_bit = Activation_bit;
		act_range_momentum = Act_range_momentum;
		running_stat = Running_stat;
		per_channel = Per_channel;
		quant_mode = Quant_mode;
		percentile = false;
		if (!per_channel)
		{
			x_min = register_buffer("x_min", torch::zeros(1));
			x_max = register_buffer("x_max", torch::zeros(1));
			act_scaling_factor = register_buffer("act_scaling_factor", torch::zeros(1));
		}
		else
		{
			assert(channel_len != NULL);
			x_min = register_buffer("x_min", torch::zeros(channel_len));
			x_max = register_buffer("x_max", torch::zeros(channel_len));
			act_scaling_factor = register_buffer("act_scaling_factor", torch::zeros(channel_len));
		}

		if (quant_mode == "none")
		{
			act_function = "";
		}
		else if (quant_mode == "symmetric")
		{
			act_function = quant_mode;//need to call SymmetricQuantFunction.apply(quant_models.py 193)
		}
		else
		{
			cout << "Unknown quantize mode: " << quant_mode << endl;
			exit(-1);
		}

	}
	torch::Tensor forward(torch::Tensor x,
		torch::Tensor pre_act_scaling_factor = torch::empty(0),
		torch::Tensor identity = torch::empty(0),
		torch::Tensor identity_scaling_factor = torch::empty(0),
		torch::Tensor specified_min = torch::empty(0),
		torch::Tensor specified_max = torch::empty(0))
	{

		torch::Tensor X_act, X_min, X_max;

		if (identity.numel() == 0)
		{
			X_act = x;
		}
		else
		{
			X_act = x + identity;
		}
		if (running_stat)
		{
			if (!percentile)
			{
				X_min = torch::min(X_act);
				X_max = torch::max(X_act);
			}
			else
			{
				cout << "need to fix" << endl;
				exit(0);
			}
		}
		else
		{
			cout << "Not implemented!" << endl;
			exit(-1);
		}

		if (X_min.equal(X_max))
		{
			x_min = x_min + X_min;
			x_max = x_max + X_max;
		}
		else if (act_range_momentum == -1)
		{
			cout << "act_range_momentum == -1 case Not implemented!" << endl;
			exit(-1);
		}
		else
		{
			x_min = x_min * act_range_momentum + X_min * (1 - act_range_momentum);
			x_max = x_max * act_range_momentum + X_max * (1 - act_range_momentum);
		}
		if (quant_mode == "none")
		{
			return X_act;
		}

		if (quant_mode != "symmetric")
		{
			printf("unspported mode!");
			exit(-1);
		}
		if (specified_min.numel() == 0)
		{
			X_min = x_min;
		}
		else
		{
			X_min = specified_min;
		}
		if (specified_max.numel() == 0)
		{
			X_max = x_max;
		}
		else
		{
			X_max = specified_max;
		}

		act_scaling_factor = symmetric_linear_quantization_params(activation_bit, x_min, x_max, per_channel);

		torch::Tensor quant_act_int;
		if (pre_act_scaling_factor.numel() == 0)
		{
			quant_act_int = SQF.forward(x, activation_bit, percentile, act_scaling_factor);
		}
		else
		{
			quant_act_int = FPM.forward(x, pre_act_scaling_factor,
				activation_bit, quant_mode,
				act_scaling_factor,
				identity, identity_scaling_factor);
		}

		auto correct_output_scale = act_scaling_factor.view(-1);
		return quant_act_int * correct_output_scale;

	}
	int activation_bit;
	double act_range_momentum;
	bool running_stat;
	bool per_channel;
	bool channel_len;
	bool percentile;
	string quant_mode;
	string act_function;
	torch::Tensor W, b;
	torch::Tensor x_min, x_max, act_scaling_factor;

	SymmetricQuantFunction SQF;
	fixedpoint_mul FPM;
	//};

	struct IntIntSoftmax : torch::nn::Module {
		IntIntSoftmax(
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

		torch::Tensor int_polynomial(torch::Tensor x_int, torch::Tensor& scaling_factor)
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

		torch::Tensor int_exp(torch::Tensor& x_int, torch::Tensor& scaling_factor)
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

		torch::Tensor forward(torch::Tensor x, torch::Tensor scaling_factor)
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

		int output_bit;
		string quant_mode;
		const double x0 = -0.9631;
		const int n = 30;
		double coef[3] = { 0.35815147, 0.96963238, 1. };
	};


	struct QuantLinear : torch::nn::Module {
		QuantLinear(
			int Weight_bit,
			int Bias_bit = NULL,
			bool Per_channel = false,
			string Quant_mode = "none")
		{
			weight_bit = Weight_bit;
			quant_mode = Quant_mode;
			per_channel = Per_channel;
			bias_bit = Bias_bit;
			quantize_bias = (bias_bit == NULL) ? false : true;
			percentile_mode = false;

			if (quant_mode != "none" && quant_mode != "symmetric")
			{
				printf("not implemented quant mode!");
				exit(-1);
			}
		}

		int set_param(torch::nn::Linear linear)
		{
			//in_features = linear.
			out_features = linear->out_features;
			fc_scaling_factor = register_buffer("fc_scaling_factor", torch::zeros(out_features));

			weight = linear->weight;
			weight_integer = register_buffer("weight_integer", torch::zeros_like(weight));
			//linear->parameters
			try
			{
				bias = linear->bias;
			}
			catch (...)
			{
				bias = torch::zeros(1);
			}
			bias_integer = register_buffer("bias_integer", torch::zeros_like(bias));
		}

		torch::Tensor forward(torch::Tensor x, torch::Tensor prev_act_scaling_factor = torch::empty(0))
		{
			if (quant_mode == "none")
			{
				return F::linear(x, weight, bias);
			}
			if (quant_mode != "symmetric")
			{
				printf("unspported mode!");
				exit(-1);
			}

			if (prev_act_scaling_factor.numel() != 0)
			{
				prev_act_scaling_factor.reshape({ 1, });
			}

			auto w = weight;
			auto w_transform = w.detach();

			torch::Tensor wmax, wmin;
			if (per_channel)
			{
				std::tuple<torch::Tensor, torch::Tensor> val = torch::max(w_transform, 1);
				wmax = std::get<0>(val);
				val = torch::min(w_transform, 1);
				wmin = std::get<0>(val);
			}
			else
			{
				wmax = w_transform.max().expand(1);
				wmin = w_transform.min().expand(1);
			}
			fc_scaling_factor = symmetric_linear_quantization_params(weight_bit, wmin, wmax, per_channel);
			weight_integer = SQF.forward(weight, weight_bit, percentile_mode, fc_scaling_factor);

			auto bias_scaling_factor = fc_scaling_factor * prev_act_scaling_factor;
			bias_integer = SQF.forward(bias, bias_bit, false, bias_scaling_factor);

			prev_act_scaling_factor = prev_act_scaling_factor.view({ 1,-1 });
			auto x_int = x / prev_act_scaling_factor;

			return F::linear(x_int, weight_integer, bias_integer) * bias_scaling_factor;//return bias_scaling_factor?
		}

		int weight_bit;
		string quant_mode;
		bool per_channel;
		int bias_bit;
		bool quantize_bias;
		bool percentile_mode;
		int in_features;
		int out_features;
		torch::Tensor weight;
		torch::Tensor bias;
		torch::Tensor fc_scaling_factor, weight_integer, bias_integer;

		SymmetricQuantFunction SQF;
	};


	struct IntLayerNorm : torch::nn::Module {
		IntLayerNorm(
			int Output_bit,
			bool Overflow_handling = true,
			string Quant_mode = "none",
			string Force_dequant = "none")
		{
			quant_mode = Quant_mode;
			if (Force_dequant == "nonlinear" || Force_dequant == "layernorm")
			{
				quant_mode = "none";
			}
			overflow_handling = Overflow_handling;
			shift = register_buffer("shift", torch::zeros(1));
			output_bit = Output_bit;
			dim_sqrt = torch::empty(0);
			//activition = QuantAct(output_bit, quant_mode);


			if (quant_mode != "none" && quant_mode != "symmetric")
			{
				printf("not implemented quant mode!");
				exit(-1);
			}
		}

		void fix()
		{
			overflow_handling = false;
		}

		void unfix()
		{
			overflow_handling = true;
		}

		int set_param(torch::nn::Linear ln)
		{
			auto normalized_shape = ln.normalized_shape;
			auto eps = ln.eps;
			auto weightcp = ln->weight.data();
			weight = weightcp.clone();
			return 0;
		}

		int set_shift(torch::Tensor y_int)
		{
			torch::NoGradGuard no_grad;
			auto y_sq_int = y_int ^ 2;
			auto var_int = torch::sum(y_sq_int, 2, true);
			auto int_log = torch::log2(torch::sqrt(var_int / (2 ^ 32))).ceil();
			auto shift2 = int_log.max();
			auto shift_old = shift;
			shift = torch::max(shift, shift2);
			cout << "Dynamic shift adjustment: " << shift_old << "->" << shift << endl;
		}

		torch::Tensor overflow_fallback(torch::Tensor y_int)
		{
			set_shift(y_int);
			auto y_int_shifted = torch::floor(y_int / (2 ^ shift));
			auto y_sq_int = y_int_shifted ^ 2;
			auto var_int = torch::sum(y_sq_int, 2, true);

			return var_int;
		}

		torch::Tensor forward(torch::Tensor x, torch::Tensor scaling_factor = torch::empty(0), torch::Tensor exponents = torch::empty(0))
		{
			if (quant_mode == "none")
			{
				auto mean = x.mean(2, true);
				auto y = x - mean;
				auto var = torch::mean(y ^ 2, 2, true);
				x = y / torch::sqrt(eps + var);
				x = x * weight + bias;
				return x;

			}
			if (quant_mode != "symmetric")
			{
				printf("unspported mode!");
				exit(-1);
			}

			if (dim_sqrt.numel() == 0)
			{
				auto n = torch::tensor(x.sizes()[2], torch::kFloat);
				dim_sqrt = torch::sqrt(n);
			}

			auto x_int = x / scaling_factor;
			auto mean_int = torch::round(x_int.mean(2, true));
			auto y_int = x_int - mean_int;
			auto y_int_shifted = torch::floor(y_int / (2 ^ shift));
			auto y_sq_int = y_int_shifted ^ 2;
			auto var_int = torch::sum(y_sq_int, 2, true);

			//overflow handling in training stage TBD

			auto std_int = torch::floor(torch::sqrt(var_int)) * (2 ^ shift);
			auto factor = torch::floor((2 ^ 31) / std_int);
			y_int = torch::floor(y_int * factor / 2);
			scaling_factor = dim_sqrt / (2 ^ 30);

			auto bias2 = bias.detach() / weight.detach();
			auto bias_int = torch::floor(bias / scaling_factor);

			y_int = y_int + bias_int;
			scaling_factor = scaling_factor * weight;
			x = y_int * scaling_factor;

			return x;
		}

		int output_bit;
		string quant_mode;
		bool overflow_handling;

		torch::Tensor shift, dim_sqrt;
		torch::Tensor weight, bias;

	};

	/*int main()
	{

		torch::Tensor specified_scale = torch::zeros({ 1 });
		cout << specified_scale << endl;
		SymmetricQuantFunction SQF1;

		return 0;
	}*/

