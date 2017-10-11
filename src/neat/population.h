#pragma once

#include <neat.h>

#include "species.h"
#include "genome.h"

struct neat_pop{
	struct neat_config conf;

	bool solved;

	struct neat_genome **genomes;
	size_t ngenomes;

	struct neat_species **species;
	size_t nspecies;

	int innovation;
};
