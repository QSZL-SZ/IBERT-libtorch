#include <QuantLinear.h>

QuantLinear::QuantLinear(
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

int QuantLinear::set_param(torch::nn::Linear linear)
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

torch::Tensor QuantLinear::forward(torch::Tensor x, torch::Tensor prev_act_scaling_factor = torch::empty(0))
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