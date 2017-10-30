#pragma once

#include <stdlib.h>
#include <stdbool.h>

enum nn_activation{
	NN_ACTIVATION_PASSTHROUGH = 0,
	NN_ACTIVATION_SIGMOID,
	NN_ACTIVATION_FAST_SIGMOID,
	NN_ACTIVATION_RELU,
	
	_NN_ACTIVATION_COUNT
};

struct nn_ffnet{
	size_t ninputs, nhiddens, noutputs, nhidden_layers;
	size_t nweights, nneurons, nactivations;

	float *weight, *output;
	char *activation;

	float bias;
};

/* Create a new feedforward neural net, the bias is set to -1.0 by default
 * input_count: 	amount of input nodes
 * hidden_count:	amount of hidden nodes per layer
 * output_count:	amount of output nodes
 * hidden_layer_count:	amount of layers for the hidden nodes
 *
 * return an allocated struct, call nn_ffnet_destroy to free it
 */
struct nn_ffnet *nn_ffnet_create(size_t input_count,
				 size_t hidden_count,
				 size_t output_count,
				 size_t hidden_layer_count);

/* Copy the feedforward network into a newly allocated one */
struct nn_ffnet *nn_ffnet_copy(struct nn_ffnet *net);

/* Deallocate the memory of the feedforward network */
void nn_ffnet_destroy(struct nn_ffnet *net);

/* Add a new hidden layer, this reallocates the memory 
 * weight	the value of a single weight between the newly created hidden
 * 		layer and the previous layer, only one weight between them is
 * 		assigned
 *
 * return a new pointer because an internal realloc is used, you should
 * overwrite the pointer you were using with this, example:
 * net = nn_ffnet_add_hidden_layer(net);
 */
struct nn_ffnet *nn_ffnet_add_hidden_layer(struct nn_ffnet *net, float weight);

/* Set the activation functions
 * hidden:	for the hidden layers
 * output:	for the output layers
 */
void nn_ffnet_set_activations(struct nn_ffnet *net,
			      enum nn_activation hidden,
			      enum nn_activation output);

/* Set the multiplier of the bias nodes, all hidden layers have 1 bias node
 * bias:	value to multiple the weight going to the bias nodes with
 */
void nn_ffnet_set_bias(struct nn_ffnet *net, float bias);

void nn_ffnet_set_weights(struct nn_ffnet *net, float weight);

/* Give all the weights in the feedforward network a value between -1 & 1 */
void nn_ffnet_randomize(struct nn_ffnet *net);

/* Run the input on the feedforward algorithm to calculate the output
 * inputs:	array of input values, assumed to be the same amount as
 * 		input_count as supplied to the nn_ffnet_create function
 *
 * return the outputs as an array of floats, the length of the array is
 * output_count as supplied to the nn_ffnet_create function
 */
float *nn_ffnet_run(struct nn_ffnet *net, const float *inputs);

bool nn_ffnet_neuron_is_connected(struct nn_ffnet *net, size_t neuron_id);

size_t nn_ffnet_get_weight_to_neuron(struct nn_ffnet *net, size_t neuron_id);
