#include "population.h"

#include <assert.h>

neat_pop_t neat_population_create(struct neat_config config)
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

	return p;
}

void neat_population_destroy(neat_pop_t population)
{
	struct neat_pop *p = population;
	assert(p);

	nn_ffnet_destroy(p->initial_genome);
	free(p);

	p = NULL;
}
