#include "network.h"

#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "population.h"

static inline double neat_fast_sigmoid(double x)
{
	return x / (1 + fabs(x));
}

static void neat_neuron_reset(struct neat_neuron *neuron)
{
	assert(neuron);

	neuron->received_inputs = 0;
	neuron->input = 0.0;
	neuron->sent_output = false;
}

static size_t neat_neuron_get_expected_inputs(struct neat_neuron neuron)
{
	if(neuron.type == NEAT_NEURON_INPUT){
		return 1;
	}

	return neuron.ninput_genes;
}

static bool neat_neuron_is_ready(struct neat_neuron neuron)
{
	size_t expected_inputs = neat_neuron_get_expected_inputs(neuron);
	bool received_all_inputs = neuron.received_inputs == expected_inputs;

	return !neuron.sent_output && received_all_inputs;
}

static void neat_neuron_add_input(struct neat_neuron *neuron, double input)
{
	neuron->input += input;
	neuron->received_inputs++;
}

static void neat_neuron_fire(struct neat_ffnet *net,
			     struct neat_neuron *neuron)
{
	assert(net);
	assert(neuron);

	neuron->sent_output = true;
	for(int i = 0; i < neuron->noutput_genes; i++){
		assert(neuron->output_genes + i);
		size_t gene_index = neuron->output_genes[i];
		assert(gene_index < net->ngenes);
		struct neat_gene *g = net->genes + gene_index;
		assert(g);

		struct neat_neuron *output = net->neurons + g->neuron_output;
		assert(output);

		if(g->enabled){
			neat_neuron_add_input(output, 0);
		}else{
			double activation = neat_fast_sigmoid(neuron->input);
			neat_neuron_add_input(output, activation * g->weight);
		}
	}
}

static void neat_neuron_add_input_gene(struct neat_ffnet *net,
				       size_t neuron_offset,
				       size_t gene_offset)
{
	assert(net);

	struct neat_neuron *n = net->neurons + neuron_offset;
	assert(n);

	size_t bytes = sizeof(size_t) * (n->ninput_genes + 1);
	n->input_genes = realloc(n->input_genes, bytes);
	assert(n->input_genes);

	n->input_genes[n->ninput_genes] = gene_offset;
	n->ninput_genes++;
}

static void neat_neuron_add_output_gene(struct neat_ffnet *net,
					size_t neuron_offset,
					size_t gene_offset)
{
	assert(net);

	struct neat_neuron *n = net->neurons + neuron_offset;
	assert(n);

	size_t bytes = sizeof(size_t) * (n->noutput_genes + 1);
	n->output_genes = realloc(n->output_genes, bytes);
	assert(n->output_genes);

	n->output_genes[n->noutput_genes] = gene_offset;
	n->noutput_genes++;
}

static struct neat_neuron *neat_ffnet_add_neuron(struct neat_ffnet *net, int id,
						 enum neat_neuron_type type)
{
	assert(net);

	size_t bytes = sizeof(struct neat_neuron) * (net->nneurons + 1);
	net->neurons = realloc(net->neurons, bytes);
	assert(net->neurons);

	struct neat_neuron *neuron = net->neurons + net->nneurons;
	*neuron = (struct neat_neuron){
		.id = id,
		.type = type,

		.input = 0.0,
		.received_inputs = 0,
		.sent_output = false
	};

	net->nneurons++;

	return neuron;
}

static struct neat_gene *neat_ffnet_add_gene(struct neat_ffnet *net,
				      	     int neuron_input_offset,
				      	     int neuron_output_offset,
				      	     double weight, bool enabled)
{
	assert(net);

	size_t bytes = sizeof(struct neat_gene) * (net->ngenes + 1);
	net->genes = realloc(net->genes, bytes);
	assert(net->genes);

	struct neat_gene *gene = net->genes + net->ngenes;
	*gene = (struct neat_gene){
		.neuron_input = neuron_input_offset,
		.neuron_output = neuron_output_offset,
		.weight = weight,
		.enabled = enabled
	};

	neat_neuron_add_input_gene(net, neuron_input_offset, net->ngenes);
	neat_neuron_add_output_gene(net, neuron_output_offset, net->ngenes);

	net->ngenes++;

	return gene;
}

struct neat_ffnet neat_ffnet_create(struct neat_config config)
{
	int ninputs = config.input_genome_topo;
	int noutputs = config.output_genome_topo;

	struct neat_ffnet net = {
		.species_id = -1,
		.generation = 0,
		.fitness = 0,
		.ngenes = 0,

		.output_offset = ninputs,
		.hidden_offset = ninputs + noutputs
	};

	int neuron_id = 0;
	for(int i = 0; i < ninputs; i++){
		neat_ffnet_add_neuron(&net, neuron_id, NEAT_NEURON_INPUT);
		neuron_id++;
	}

	for(int i = 0; i < noutputs; i++){
		neat_ffnet_add_neuron(&net, neuron_id, NEAT_NEURON_OUTPUT);
		neuron_id++;
	}

	for(int i = 0; i < ninputs; i++){
		for(int j = 0; j < noutputs; j++){
			int index = net.output_offset + j;
			neat_ffnet_add_gene(&net, i, index, 0.0, true);
		}
	}

	neat_ffnet_randomize_weights(&net);

	return net;
}

struct neat_ffnet neat_ffnet_copy(struct neat_ffnet *src)
{
	assert(src);

	struct neat_ffnet dest;
	memcpy(&dest, src, sizeof(struct neat_ffnet));

	dest.neurons = malloc(sizeof(struct neat_neuron) * dest.nneurons);
	assert(dest.neurons);
	for(int i = 0; i < dest.nneurons; i++){
		struct neat_neuron *n_dest = dest.neurons + i;
		assert(n_dest);

		struct neat_neuron *n_src = src->neurons + i;
		assert(n_src);
		memcpy(n_dest, n_src, sizeof(struct neat_neuron));
		assert(n_dest);
		
		size_t bytes = sizeof(size_t) * n_dest->ninput_genes;
		n_dest->input_genes = malloc(bytes);
		assert(n_dest->input_genes);
		for(int j = 0; j < n_dest->ninput_genes; j++){
			memcpy(n_dest->input_genes + j,
			       n_src->input_genes + j,
			       sizeof(size_t));
		}

		bytes = sizeof(size_t) * n_dest->noutput_genes;
		n_dest->output_genes = malloc(bytes);
		assert(n_dest->output_genes);
		for(int j = 0; j < n_dest->noutput_genes; j++){
			memcpy(n_dest->output_genes + j,
			       n_src->output_genes + j,
			       sizeof(size_t));
		}
	}

	dest.genes = malloc(sizeof(struct neat_gene) * dest.ngenes);
	assert(dest.genes);
	for(int i = 0; i < dest.ngenes; i++){
		dest.genes[i] = src->genes[i];
	}

	return dest;
}

void neat_ffnet_destroy(struct neat_ffnet *net)
{
	assert(net);

	for(int i = 0; i < net->nneurons; i++){
		free(net->neurons[i].input_genes);
		free(net->neurons[i].output_genes);
	}
	free(net->neurons);
	free(net->genes);
	free(net);

	net = NULL;
}

void neat_ffnet_randomize_weights(struct neat_ffnet *net)
{
	assert(net);

	for(int i = 0; i < net->ngenes; i++){
		assert(net->genes + i);

		/* Random double between -2 and 2 */
		double random = (double)rand() / RAND_MAX * 4.0 - 2.0;
		net->genes[i].weight = random;
	}
}

void neat_ffnet_mutate(struct neat_ffnet *net)
{
	/* TODO implement mutation */
}

inline size_t neat_ffnet_get_input_size(struct neat_ffnet *net)
{
	assert(net);

	return net->output_offset;
}

inline size_t neat_ffnet_get_output_size(struct neat_ffnet *net)
{
	assert(net);

	return net->hidden_offset - net->output_offset;
}

inline size_t neat_ffnet_get_hidden_size(struct neat_ffnet *net)
{
	assert(net);

	return net->nneurons - net->hidden_offset;
}

void neat_ffnet_predict(neat_ffnet_t network, const double *inputs)
{
	struct neat_ffnet *net = network;
	assert(net);

	size_t ninputs = neat_ffnet_get_input_size(net);

	/* TODO scale inputs */
	for(int i = 0; i < ninputs; i++){
		neat_neuron_add_input(net->neurons + i, inputs[i]);
	}

	bool finished = false;
	while(!finished){
		finished = true;
		
		for(int i = 0; i < net->nneurons; i++){
			struct neat_neuron *neuron = net->neurons + i;
			assert(neuron);
			if(neat_neuron_is_ready(*neuron)){
				neat_neuron_fire(net, neuron);
			}

			if(!neuron->sent_output){
				finished = false;
			}
		}
	}
}

double *neat_ffnet_get_outputs(neat_ffnet_t network)
{	
	struct neat_ffnet *net = network;
	assert(net);
	struct neat_neuron *ns = net->neurons + net->output_offset;
	assert(ns);

	size_t noutputs = net->hidden_offset - net->output_offset;
	double *outputs = malloc(sizeof(double) * noutputs);
	assert(outputs);

	for(int i = 0; i < noutputs; i++){
		assert(ns + i);
		outputs[i] = ns[i].input;
	}

	return outputs;
}

double neat_ffnet_get_output(neat_ffnet_t network, size_t index)
{
	struct neat_ffnet *net = network;
	assert(net);
	struct neat_neuron *ns = net->neurons + net->output_offset;
	assert(ns);

	assert(ns + index);

	return ns[index].input;
}

void neat_ffnet_reset(neat_ffnet_t network)
{
	struct neat_ffnet *net = network;
	assert(net);

	for(int i = 0; i < net->nneurons; i++){
		neat_neuron_reset(net->neurons + i);
	}
}
