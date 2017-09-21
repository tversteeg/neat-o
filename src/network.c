#include "network.h"

#include <stdlib.h>

#include "population.h"

static void neat_neuron_add_input_gene(struct neat_neuron *neuron,
				       struct neat_gene *input)
{
	size_t bytes = sizeof(struct neat_gene) * (neuron->ninput_genes + 1);
	neuron->input_genes = realloc(neuron->input_genes, bytes);

	neuron->input_genes[neuron->ninput_genes] = input;
	neuron->ninput_genes++;
}

static void neat_neuron_add_output_gene(struct neat_neuron *neuron,
					struct neat_gene *output)
{
	size_t bytes = sizeof(struct neat_gene) * (neuron->noutput_genes + 1);
	neuron->output_genes = realloc(neuron->output_genes, bytes);

	neuron->output_genes[neuron->noutput_genes] = output;
	neuron->noutput_genes++;
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

struct neat_gene *neat_ffnet_add_gene(struct neat_ffnet *net,
				      struct neat_neuron *input,
				      struct neat_neuron *output,
				      double weight, bool enabled)
{
	size_t bytes = sizeof(struct neat_gene) * (net->ngenes + 1);
	net->genes = realloc(net->genes, bytes);

	struct neat_gene *gene = net->genes + net->ngenes;
	*gene = (struct neat_gene){
		.input = input,
		.output = output,
		.weight = weight,
		.enabled = enabled
	};

	neat_neuron_add_input_gene(input, gene);
	neat_neuron_add_output_gene(output, gene);

	net->ngenes++;

	return gene;
}

neat_ffnet_t neat_ffnet_create(neat_pop_t population)
{
	struct neat_pop *p = population;

	int ninputs = p->conf.input_genome_topo;
	int noutputs = p->conf.output_genome_topo;

	struct neat_ffnet *net = malloc(sizeof(struct neat_ffnet));

	int neuron_id = 0;
	int i;
	for(i = 0; i < ninputs; i++){
		neat_ffnet_add_neuron(net, neuron_id++, NEAT_NEURON_INPUT);
	}
	for(i = 0; i < noutputs; i++){
		neat_ffnet_add_neuron(net, neuron_id++, NEAT_NEURON_OUTPUT);
	}

	for(i = 0; i < ninputs; i++){
		struct neat_neuron *input = net->neurons + i;

		int j;
		for(j = 0; j < noutputs; j++){
			struct neat_neuron *output = net->neurons + ninputs + i;

			neat_ffnet_add_gene(net, input, output, 0.0, true);
		}
	}

	return net;
}

neat_ffnet_t neat_ffnet_activate(neat_ffnet_t net, double* inputs, int ninputs)
{
	return NULL;
}

int neat_ffnet_get_outputs(neat_ffnet_t net, double **outputs)
{
	return 0;
}

double neat_ffnet_get_output_at_index(neat_ffnet_t net, int index)
{
	return 0.0;
}
