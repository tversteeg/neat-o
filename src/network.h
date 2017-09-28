#pragma once

#include <neat.h>

#include "gene.h"
#include "neuron.h"

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
void neat_ffnet_copynew(struct neat_ffnet *dst, struct neat_ffnet *src);
void neat_ffnet_copy(struct neat_ffnet *dst, struct neat_ffnet *src);
void neat_ffnet_destroy(struct neat_ffnet *net);

void neat_ffnet_randomize_weights(struct neat_ffnet *net);
void neat_ffnet_mutate(struct neat_ffnet *net);

size_t neat_ffnet_get_input_size(struct neat_ffnet *net);
size_t neat_ffnet_get_output_size(struct neat_ffnet *net);
size_t neat_ffnet_get_hidden_size(struct neat_ffnet *net);
