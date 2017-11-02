#pragma once

#include <neat.h>
#include <nn.h>

#include "species.h"

struct neat_genome{
	struct nn_ffnet *net;
	int *innovations;
	size_t ninnovations;
	size_t used_weights;

	float fitness;
	size_t time_alive;
};

struct neat_genome *neat_genome_create(struct neat_config config,
				       int innovation);
struct neat_genome *neat_genome_copy(const struct neat_genome *genome);
struct neat_genome *neat_genome_reproduce(const struct neat_genome *parent1,
					  const struct neat_genome *parent2);
void neat_genome_destroy(struct neat_genome *genome);

const float *neat_genome_run(struct neat_genome *genome, const float *inputs);

void neat_genome_mutate(struct neat_genome *genome,
			struct neat_config config,
			int innovation);

bool neat_genome_is_compatible(const struct neat_genome *genome,
			       const struct neat_genome *other,
			       float treshold,
			       size_t total_species);

void neat_genome_print_net(const struct neat_genome *genome);
