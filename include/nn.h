#pragma once

#include <stdlib.h>

enum nn_activation{
	NN_ACTIVATION_SIGMOID,
	NN_ACTIVATION_FAST_SIGMOID,
	NN_ACTIVATION_RELU
};

struct nn_ffnet{
	size_t ninputs, nhiddens, noutputs, nhidden_layers;
	size_t nweights, nneurons;

	float *weight, *output;

	float bias;

	enum nn_activation hidden_activation, output_activation;
};

/* Create a new feedforward neural net, the activation functions for the hidden
 * and output layers are set by default to NN_ACTIVATION_SIGMOID
 * input_count: 	maximum amount of input nodes
 * hidden_count:	maximum amount of hidden nodes per layer
 * output_count:	maximum amount of output nodes
 * hidden_layer_count:	amount of layers for the hidden nodes, total amount of
 * 			hidden nodes is hidden_count * hidden_layer_count
 */
struct nn_ffnet *nn_ffnet_create(size_t input_count,
				 size_t hidden_count,
				 size_t output_count,
				 size_t hidden_layer_count);

/* Copy the feedforward network into a newly allocated one */
struct nn_ffnet *nn_ffnet_copy(struct nn_ffnet *net);

/* Deallocate the memory of the feedforward network */
void nn_ffnet_destroy(struct nn_ffnet *net);

/* Set the activation functions
 * hidden:	for the hidden layers
 * output:	for the output layers
 */
void nn_ffnet_set_activations(struct nn_ffnet *net,
			      enum nn_activation hidden,
			      enum nn_activation output);

void nn_ffnet_set_bias(struct nn_ffnet *net, float bias);

/* Give all the weights in the feedforward network a value between -1 & 1 */
void nn_ffnet_randomize(struct nn_ffnet *net);

/* Run the input on the feedforward algorithm to calculate the output
 * inputs:	array of input values, assumed to be the same amount as
 * 		input_count as supplied to the nn_ffnet_create function
 *
 * returns the outputs as an array of floats, the length of the array is
 * output_count as supplied to the nn_ffnet_create function
 */
float *nn_ffnet_run(struct nn_ffnet *net, const float *inputs);
