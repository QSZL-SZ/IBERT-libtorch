#pragma once
#ifndef _QUANTACT_H_
#define _QUANTACT_H_

#include <quant_utils.h>

using namespace std;

class QuantAct : torch::nn::Module {
public:
	QuantAct(int Activation_bit,
		string Quant_mode = "none",
		double Act_range_momentum = 0.95,
		bool Running_stat = true,
		bool Per_channel = false,
		bool channel_len = NULL);

	torch::Tensor forward(torch::Tensor x,
		torch::Tensor pre_act_scaling_factor = torch::empty(0),
		torch::Tensor identity = torch::empty(0),
		torch::Tensor identity_scaling_factor = torch::empty(0),
		torch::Tensor specified_min = torch::empty(0),
		torch::Tensor specified_max = torch::empty(0));

private:
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
};

#endif