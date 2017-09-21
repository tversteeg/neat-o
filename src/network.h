#pragma once

#include <neat.h>

enum neat_neuron_type{
	NEAT_NEURON_INPUT,
	NEAT_NEURON_HIDDEN,
	NEAT_NEURON_OUTPUT
};

struct neat_neuron;

struct neat_gene{
	struct neat_neuron *input, *output;

	bool enabled;
	double weight;
};

struct neat_neuron{
	int id;
	enum neat_neuron_type type;

	struct neat_gene **input_genes, **output_genes;
	int ninput_genes, noutput_genes;

	double input;
};

struct neat_ffnet{
	struct neat_neuron *neurons, *hidden_neurons;
	int nneurons, nhidden_neurons;

	struct neat_gene *genes;
	int ngenes;
};
