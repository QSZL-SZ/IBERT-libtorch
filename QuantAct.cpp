#include <QuantAct.h>

QuantAct::QuantAct(int Activation_bit,
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
torch::Tensor QuantAct::forward(torch::Tensor x,
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