#pragma once
#ifndef _QUANTLINEAR_H_
#define _QUANTLINEAR_H_

#include <quant_utils.h>

class QuantLinear : torch::nn::Module {
public:
	QuantLinear(
		int Weight_bit,
		int Bias_bit = NULL,
		bool Per_channel = false,
		string Quant_mode = "none");

	int set_param(torch::nn::Linear linear);

	torch::Tensor forward(torch::Tensor x, torch::Tensor prev_act_scaling_factor = torch::empty(0));
private:
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

#endif