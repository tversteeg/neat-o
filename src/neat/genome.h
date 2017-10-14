#pragma once

#include <neat.h>
#include <nn.h>

#include "species.h"

struct neat_genome{
	struct nn_ffnet *net;
	int *innovations;
	float fitness;
	int time_alive;
};

struct neat_genome *neat_genome_create(struct neat_config config,
				       int innovation);
struct neat_genome *neat_genome_copy(const struct neat_genome *genome);
void neat_genome_destroy(struct neat_genome *genome);

const float *neat_genome_run(struct neat_genome *genome, const float *inputs);

void neat_genome_add_random_node(struct neat_genome *genome, int innovation);

bool neat_genome_is_compatible(const struct neat_genome *genome,
			       const struct neat_genome *other,
			       float treshold);
