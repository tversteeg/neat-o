#pragma once

#include <neat.h>
#include <nn.h>

struct neat_species{
	bool active;

	struct nn_ffnet **genomes;
	size_t ngenomes;
};

struct neat_species *neat_species_create(struct neat_config config,
					 struct nn_ffnet *genome);

void neat_species_destroy(struct neat_species *species);
