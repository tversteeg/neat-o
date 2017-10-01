#pragma once

#include <stdlib.h>

enum nn_activation{
	NN_ACTIVATION_SIGMOID
};

struct nn_ffnet{
	size_t ninputs, nhiddens, noutputs, hidden_layer_count;
	size_t nweights, nneurons;

	double *weight, *output, *delta;

	enum nn_activation hidden_activation, output_activation;
};

struct nn_ffnet *nn_ffnet_create(size_t input_count,
				 size_t hidden_count,
				 size_t output_count,
				 size_t hidden_layer_count);
void nn_ffnet_destroy(struct nn_ffnet *net);

void nn_ffnet_randomize(struct nn_ffnet *net);
