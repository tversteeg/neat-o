#include <nn.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

struct nn_ffnet *nn_ffnet_create(size_t input_count,
				 size_t hidden_count,
				 size_t output_count,
				 size_t hidden_layer_count)
{
	assert(input_count > 0);
	assert(output_count > 0);

	size_t hidden_weights = 0;
	if(hidden_layer_count > 0){
		size_t input_weights = (input_count + 1) * hidden_count;
		size_t hidden_internal_weights = (hidden_layer_count - 1) *
						 (hidden_count + 1) *
						 hidden;
		hidden_weights = input_weights + hidden_internal_weights;
	}

	size_t output_weights = 0;
	if(hidden_layer_count > 0){
		output_weights = hidden_count + 1;
	}else{
		output_weights = input_count + 1;
	}
	output_weights *= output_count;

	size_t total_weights = hidden_weights + output_weights;
	size_t total_neurons = input_count +
			       hidden_count * hidden_layer_count +
			       output_count;

	size_t total_deltas = total_neurons - input_count;

	size_t total_items = total_weights + total_neurons + total_deltas;

	/* Allocate the struct with extra bytes behind it for the data */
	size_t bytes = sizeof(struct nn_ffnet) + sizeof(double) * total_items;
	struct nn_ffnet *net = malloc(bytes);
	assert(net);

	net->ninputs = input_count;
	net->nhiddens = hidden_count;
	net->noutputs = output_count;
	net->hidden_layer_count = hidden_layer_count;

	net->nweights = total_weights;
	net->nneurons = total_neurons;

	/* Point the pointers to the extra allocated data */
	net->weight = (double*)((char*)net + sizeof(struct net_ffnet));
	net->output = net->weight + net->nweights;
	net->delta = net->weight + net->nneurons;

	net->hidden_activation = NN_ACTIVATION_SIGMOID;
	net->output_activation = NN_ACTIVATION_SIGMOID;

	return net;
}
