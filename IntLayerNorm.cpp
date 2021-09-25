#include <IntLayerNorm.h>

IntLayerNorm::IntLayerNorm(
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

void IntLayerNorm::fix()
{
	overflow_handling = false;
}

void IntLayerNorm::unfix()
{
	overflow_handling = true;
}

int IntLayerNorm::set_param(torch::nn::Linear ln)
{
	auto normalized_shape = ln.normalized_shape;
	auto eps = ln.eps;
	auto weightcp = ln->weight.data();
	weight = weightcp.clone();
	return 0;
}

int IntLayerNorm::set_shift(torch::Tensor y_int)
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

torch::Tensor IntLayerNorm::overflow_fallback(torch::Tensor y_int)
{
	set_shift(y_int);
	auto y_int_shifted = torch::floor(y_int / (2 ^ shift));
	auto y_sq_int = y_int_shifted ^ 2;
	auto var_int = torch::sum(y_sq_int, 2, true);

	return var_int;
}

torch::Tensor IntLayerNorm::forward(torch::Tensor x, torch::Tensor scaling_factor = torch::empty(0), torch::Tensor exponents = torch::empty(0))
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