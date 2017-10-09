#pragma once

#include <neat.h>
#include <nn.h>

#include "species.h"

struct neat_pop{
	struct neat_config conf;

	bool solved;

	struct nn_ffnet **organisms;
	size_t norganisms;

	struct neat_species **species;
	size_t nspecies;
};
