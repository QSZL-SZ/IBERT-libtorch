#pragma once

#ifndef _INTINTSOFTMAX_H_
#define _INTINTSOFTMAX_H_

#include <torch/torch.h>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

class IntIntSoftmax : torch::nn::Module {
	IntIntSoftmax(
		int Output_bit,
		string Quant_mode = "none",
		string Force_dequant = "none");

	torch::Tensor int_polynomial(torch::Tensor x_int, torch::Tensor& scaling_factor);
	

	torch::Tensor int_exp(torch::Tensor& x_int, torch::Tensor& scaling_factor);

	torch::Tensor forward(torch::Tensor x, torch::Tensor scaling_factor);

private:
	int output_bit;
	string quant_mode;
	const double x0 = -0.9631;
	const int n = 30;
	double coef[3] = { 0.35815147, 0.96963238, 1. };
};

#endif