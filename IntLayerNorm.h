#pragma once
#ifndef _INTLAYERNORM_H_
#define _INTLAYERNORM_H_

#include <quant_utils.h>


class IntLayerNorm : torch::nn::Module {
public:
	IntLayerNorm(
		int Output_bit,
		bool Overflow_handling = true,
		string Quant_mode = "none",
		string Force_dequant = "none");

	void fix();

	void unfix();

	int set_param(torch::nn::Linear ln);

	int set_shift(torch::Tensor y_int);

	torch::Tensor overflow_fallback(torch::Tensor y_int);

	torch::Tensor forward(torch::Tensor x, torch::Tensor scaling_factor = torch::empty(0), torch::Tensor exponents = torch::empty(0));
private:
	int output_bit;
	string quant_mode;
	bool overflow_handling;

	torch::Tensor shift, dim_sqrt;
	torch::Tensor weight, bias;

};

#endif