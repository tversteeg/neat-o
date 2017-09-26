#pragma once

#include <neat.h>

#include "network.h"
#include "species.h"

struct neat_pop{
	struct neat_config conf;

	bool solved;

	struct neat_ffnet *initial_genome;

	struct neat_species *species;
	int nspecies;
};
