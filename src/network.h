#pragma once

#include <neat.h>

enum neat_neuron_type{
	NEAT_NEURON_INPUT,
	NEAT_NEURON_HIDDEN,
	NEAT_NEURON_OUTPUT
};

struct neat_neuron;

struct neat_gene{
	size_t neuron_input, neuron_output;

	bool enabled;
	double weight;
};

struct neat_neuron{
	int id;
	enum neat_neuron_type type;

	size_t *input_genes, *output_genes;
	size_t ninput_genes, noutput_genes;

	double input;
	size_t received_inputs;
	bool sent_output;
};

struct neat_ffnet{
	/* The order of the neurons array is: [input, output, hidden] */
	struct neat_neuron *neurons;
	size_t nneurons, output_offset, hidden_offset;

	struct neat_gene *genes;
	size_t ngenes;

	int species_id, generation;

	double fitness;
};

struct neat_ffnet neat_ffnet_create(struct neat_config config);
struct neat_ffnet neat_ffnet_copy(struct neat_ffnet *net);

void neat_ffnet_randomize_weights(struct neat_ffnet *net);

size_t neat_ffnet_get_input_size(struct neat_ffnet *net);
size_t neat_ffnet_get_output_size(struct neat_ffnet *net);
size_t neat_ffnet_get_hidden_size(struct neat_ffnet *net);
