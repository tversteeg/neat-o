#include "network.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "population.h"

static struct neat_neuron *neat_ffnet_add_neuron(struct neat_ffnet *net,
						 enum neat_neuron_type type)
{
	assert(net);

	if(net->nneurons == 0){
		net->neurons = malloc(sizeof(struct neat_neuron));
	}else{
		size_t bytes = sizeof(struct neat_neuron) * (net->nneurons + 1);
		net->neurons = realloc(net->neurons, bytes);
	}
	assert(net->neurons);

	struct neat_neuron *neuron = net->neurons + net->nneurons;
	*neuron = (struct neat_neuron){
		.id = net->nneurons,
		.type = type,

		.input = 0.0,
		.received_inputs = 0,
		.sent_output = false
	};

	net->nneurons++;

	return neuron;
}

static void neat_ffnet_add_gene(struct neat_ffnet *net,
				int neuron_input_offset,
				int neuron_output_offset,
				double weight,
				bool enabled)
{
	assert(net);

	if(net->ngenes == 0){
		net->genes = malloc(sizeof(struct neat_gene));
	}else{
		size_t bytes = sizeof(struct neat_gene) * (net->ngenes + 1);
		net->genes = realloc(net->genes, bytes);
	}
	assert(net->genes);

	struct neat_gene *gene = net->genes + net->ngenes;
	*gene = (struct neat_gene){
		.neuron_input = neuron_input_offset,
		.neuron_output = neuron_output_offset,
		.weight = weight,
		.enabled = enabled
	};

	neat_neuron_add_output_gene(net, neuron_input_offset, net->ngenes);
	neat_neuron_add_input_gene(net, neuron_output_offset, net->ngenes);

	++net->ngenes;
}

struct neat_ffnet neat_ffnet_create(struct neat_config config)
{
	int ninputs = config.input_genome_topo;
	int noutputs = config.output_genome_topo;

	struct neat_ffnet net = {
		.species_id = -1,
		.generation = 0,
		.fitness = 0,

		.nneurons = 0,
		.ngenes = 0,

		.output_offset = ninputs,
		.hidden_offset = ninputs + noutputs
	};

	for(int i = 0; i < ninputs; i++){
		neat_ffnet_add_neuron(&net, NEAT_NEURON_INPUT);
	}

	for(int i = 0; i < noutputs; i++){
		neat_ffnet_add_neuron(&net, NEAT_NEURON_OUTPUT);
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

void neat_ffnet_copynew(struct neat_ffnet *dst, struct neat_ffnet *src)
{
	assert(src);

	memcpy(dst, src, sizeof(struct neat_ffnet));

	size_t bytes = sizeof(struct neat_gene) * dst->ngenes;
	if(bytes > 0){
		dst->genes = malloc(bytes);
		assert(dst->genes);

		memcpy(dst->genes, src->genes, bytes);
	}else{
		dst->genes = NULL;
	}

	bytes = sizeof(struct neat_neuron) * dst->nneurons;
	if(bytes == 0){
		dst->neurons = NULL;
		return;
	}
	dst->neurons = malloc(bytes);
	assert(dst->neurons);
	memcpy(dst->neurons, src->neurons, bytes);

	for(int i = 0; i < dst->nneurons; i++){
		struct neat_neuron *n_dst = dst->neurons + i;
		struct neat_neuron *n_src = src->neurons + i;
		
		bytes = sizeof(size_t) * n_dst->ninput_genes;
		if(bytes > 0){
			n_dst->input_genes = malloc(bytes);
			assert(n_dst->input_genes);
			memcpy(n_dst->input_genes, n_src->input_genes, bytes);
		}else{
			n_dst->input_genes = NULL;
		}

		bytes = sizeof(size_t) * n_dst->noutput_genes;
		if(bytes > 0){
			n_dst->output_genes = malloc(bytes);
			assert(n_dst->output_genes);
			memcpy(n_dst->output_genes, n_src->output_genes, bytes);
		}else{
			n_dst->output_genes = NULL;
		}
	}
}

void neat_ffnet_copy(struct neat_ffnet *dst, struct neat_ffnet *src)
{
	assert(dst);
	assert(src);

	if(dst->ngenes > 0){
		free(dst->genes);
	}
	if(dst->nneurons > 0){
		for(int i = 0; i < dst->nneurons; i++){
			if(dst->neurons[i].ninput_genes > 0){
				free(dst->neurons[i].input_genes);
			}
			if(dst->neurons[i].noutput_genes > 0){
				free(dst->neurons[i].output_genes);
			}
		}
		free(dst->neurons);
	}

	neat_ffnet_copynew(dst, src);
}

void neat_ffnet_destroy(struct neat_ffnet *net)
{
	assert(net);

	if(net->nneurons > 0){
		for(int i = 0; i < net->nneurons; i++){
			if(net->neurons[i].ninput_genes > 0){
				free(net->neurons[i].input_genes);
			}
			if(net->neurons[i].noutput_genes > 0){
				free(net->neurons[i].output_genes);
			}
		}
		free(net->neurons);
		net->neurons = NULL;
		net->nneurons = 0;
	}
	if(net->ngenes > 0){
		free(net->genes);
		net->genes = NULL;
		net->ngenes = 0;
	}
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
	assert(net);

	for(int i = 0; i < net->ngenes; i++){
		neat_gene_mutate_weight(net->genes + i);
	}

	/* TODO change to config.ADD_GENE_MUTATION_RATE */
	if(rand() < RAND_MAX / 20){
		bool gene_added = false;
		while(!gene_added){
			if(neat_ffnet_get_hidden_size(net) < 1){
				break;
			}

			break;
		}
	}

	/* TODO change to config.ADD_NODE_MUTATION_RATE */
	if(rand() < RAND_MAX / 33){
		size_t gene_num = rand() % net->ngenes;
		struct neat_gene *selected = net->genes + gene_num;
		assert(selected);

		if(!selected->enabled){
			return;
		}

		printf("Add node mutation for gene %d\n", gene_num);

		selected->enabled = false;

		struct neat_neuron *neuron;
		neuron = neat_ffnet_add_neuron(net, NEAT_NEURON_HIDDEN);

		printf("%d ->(1.0) %d\n", selected->neuron_input, neuron->id);
		neat_ffnet_add_gene(net, selected->neuron_input, neuron->id,
				    1.0, true);

		printf("%d ->(%g) %d\n", neuron->id, selected->neuron_output, selected->weight);
		neat_ffnet_add_gene(net, neuron->id, selected->neuron_output,
				    selected->weight, true);
	}
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

void print_tree(struct neat_ffnet *net, struct neat_neuron *n)
{
	printf("%d(", n->id);
	for(int i = 0; i < n->ninput_genes; i++){
		struct neat_gene *gene = net->genes + n->input_genes[i];
		printf("%d,", gene->neuron_input);
	}
	printf(") -> (");
	for(int i = 0; i < n->noutput_genes; i++){
		struct neat_gene *gene = net->genes + n->output_genes[i];
		printf("%d,", gene->neuron_output);
	}
	printf(")\n");
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
			if(neuron->type == NEAT_NEURON_OUTPUT){
				continue;
			}

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
