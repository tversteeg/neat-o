#include "network.h"

#include <stdlib.h>

#include "population.h"

static void neat_neuron_add_input_gene(struct neat_ffnet *net,
				       size_t gene_offset,
				       size_t neuron_offset)
{
	struct neat_neuron *n = net->neurons + neuron_offset;

	size_t bytes = sizeof(size_t) * (n->ninput_genes + 1);
	n->input_genes = realloc(n->input_genes, bytes);

	n->input_genes[n->ninput_genes] = gene_offset;
	n->ninput_genes++;
}

static void neat_neuron_add_output_gene(struct neat_ffnet *net,
					size_t gene_offset,
					size_t neuron_offset)
{
	struct neat_neuron *n = net->neurons + neuron_offset;

	size_t bytes = sizeof(size_t) * (n->noutput_genes + 1);
	n->output_genes = realloc(n->output_genes, bytes);

	n->output_genes[n->noutput_genes] = gene_offset;
	n->noutput_genes++;
}

static struct neat_neuron *neat_ffnet_add_neuron(struct neat_ffnet *net, int id,
						 enum neat_neuron_type type)
{
	size_t bytes = sizeof(struct neat_neuron) * (net->nneurons + 1);
	net->neurons = realloc(net->neurons, bytes);

	struct neat_neuron *neuron = net->neurons + net->nneurons;
	*neuron = (struct neat_neuron){
		.id = id,
		.type = type,
		.input = 0.0
	};

	net->nneurons++;

	return neuron;
}

static struct neat_gene *neat_ffnet_add_gene(struct neat_ffnet *net,
				      	     int neuron_input_offset,
				      	     int neuron_output_offset,
				      	     double weight, bool enabled)
{
	size_t bytes = sizeof(struct neat_gene) * (net->ngenes + 1);
	net->genes = realloc(net->genes, bytes);

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

		.output_offset = ninputs,
		.hidden_offset = ninputs + noutputs
	};

	int neuron_id = 0;
	for(int i = 0; i < ninputs; i++){
		neat_ffnet_add_neuron(&net, neuron_id++, NEAT_NEURON_INPUT);
	}

	for(int i = 0; i < noutputs; i++){
		neat_ffnet_add_neuron(&net, neuron_id++, NEAT_NEURON_OUTPUT);
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
	struct neat_ffnet dest = *src;

	dest.neurons = malloc(sizeof(struct neat_neuron) * dest.nneurons);
	for(int i = 0; i < dest.nneurons; i++){
		struct neat_neuron *n_dest = dest.neurons + i;
		struct neat_neuron *n_src = src->neurons + i;
		*n_dest = *n_src;
		
		size_t bytes = sizeof(size_t) * n_dest->ninput_genes;
		n_dest->input_genes = malloc(bytes);
		for(int j = 0; j < n_dest->ninput_genes; j++){
			n_dest->input_genes[j] = n_src->input_genes[j];
		}

		bytes = sizeof(size_t) * n_dest->noutput_genes;
		n_dest->output_genes = malloc(bytes);
		for(int j = 0; j < n_dest->noutput_genes; j++){
			n_dest->output_genes[j] = n_src->output_genes[j];
		}
	}

	dest.genes = malloc(sizeof(struct neat_gene) * dest.ngenes);
	for(int i = 0; i < dest.ngenes; i++){
		dest.genes[i] = src->genes[i];
	}

	return dest;
}

void neat_ffnet_randomize_weights(struct neat_ffnet *net)
{
	for(int i = 0; i < net->ngenes; i++){
		/* Random double between -2 and 2 */
		double random = (double)rand() / RAND_MAX * 4.0 - 2.0;
		net->genes[i].weight = random;
	}
}

double *neat_ffnet_get_outputs(struct neat_ffnet *net)
{
	if(!net){
		fprintf(stderr, "Network for species %d is NULL\n",
			net->species_id);
		return NULL;
	}

	struct neat_neuron *ns = net->neurons + net->output_offset;

	size_t noutputs = net->hidden_offset - net->output_offset;
	double *outputs = malloc(noutputs * sizeof(double));

	for(int i = 0; i < noutputs; i++){
		outputs[i] = ns[i].input;
	}

	return outputs;
}
