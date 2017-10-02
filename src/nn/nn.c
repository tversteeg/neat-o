#include <nn.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

static double nn_rand(double start, double end)
{
	assert(start < end);

	double range = end - start;

	return (rand() / (double)RAND_MAX) * range + start;
}

static void nn_ffnet_set_pointers(struct nn_ffnet *net)
{
	assert(net);

	net->weight = (double*)((char*)net + sizeof(struct nn_ffnet));
	net->output = net->weight + net->nweights;
	net->delta = net->weight + net->nneurons;
}

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
						 hidden_count;
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

	net->hidden_activation = NN_ACTIVATION_SIGMOID;
	net->output_activation = NN_ACTIVATION_SIGMOID;

	nn_ffnet_set_pointers(net);

	return net;
}

struct nn_ffnet *nn_ffnet_copy(struct nn_ffnet *net)
{
	assert(net);

	size_t delta_size = (net->nneurons - net->ninputs);
	size_t extra = net->nweights + net->nneurons + delta_size;
	size_t bytes = sizeof(struct nn_ffnet) + sizeof(double) * extra;
	assert(bytes > sizeof(struct nn_ffnet));

	struct nn_ffnet *new = malloc(bytes);
	assert(new);

	memcpy(new, net, bytes);

	nn_ffnet_set_pointers(new);

	return new;
}

void nn_ffnet_destroy(struct nn_ffnet *net)
{
	assert(net);

	free(net);
}

void nn_ffnet_randomize(struct nn_ffnet *net)
{
	assert(net);

	for(int i = 0; i < net->nweights; i++){
		net->weight[i] = nn_rand(-0.5, 0.5);
	}
}

void nn_ffnet_set_activations(struct nn_ffnet *net,
			      enum nn_activation hidden,
			      enum nn_activation output)
{
	assert(net);

	net->hidden_activation = hidden;
	net->output_activation = output;
}

double *nn_ffnet_run(struct nn_ffnet *net, const double *inputs)
{
	return NULL;
}

void nn_ffnet_train(struct nn_ffnet *net,
		    const double *inputs,
		    const double *wanted_outputs,
		    double learning_rate)
{
	
}
