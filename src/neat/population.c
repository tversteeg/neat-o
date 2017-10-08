#include "population.h"

#include <assert.h>

neat_t neat_create(struct neat_config config)
{
	assert(config.population_size > 0);

	struct neat_pop *p = calloc(1, sizeof(struct neat_pop));
	assert(p);

	p->solved = false;
	p->conf = config;
	p->initial_genome = nn_ffnet_create(config.network_inputs,
					    config.network_hidden_nodes,
					    config.network_outputs,
					    config.network_hidden_layers);

	p->nspecies = 1;
	p->species = malloc(sizeof(struct neat_species*));
	assert(p->species);

	p->species[0] = neat_species_create(config, p->initial_genome);

	return p;
}

void neat_destroy(neat_t population)
{
	struct neat_pop *p = population;
	assert(p);

	nn_ffnet_destroy(p->initial_genome);
	for(size_t i = 0; i < p->nspecies; i++){
		neat_species_destroy(p->species[i]);
	}
	free(p->species);
	free(p);

	p = NULL;
}
