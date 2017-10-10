#include "population.h"

#include <float.h>
#include <assert.h>

neat_t neat_create(struct neat_config config)
{
	assert(config.population_size > 0);

	struct neat_pop *p = calloc(1, sizeof(struct neat_pop));
	assert(p);

	p->solved = false;
	p->conf = config;

	/* Allocate memory for all the organisms */
	p->norganisms = config.population_size;
	p->organisms = malloc(sizeof(struct nn_ffnet*) *
			       config.population_size);
	p->organism_fitness = calloc(config.population_size, sizeof(float));
	p->organism_time_alive = calloc(config.population_size, sizeof(int));
	assert(p->norganisms);

	/* Create a base organism and copy it for every other one */
	p->organisms[0] = nn_ffnet_create(config.network_inputs,
					  config.network_hidden_nodes,
					  config.network_outputs,
					  config.network_hidden_layers);

	nn_ffnet_set_activations(p->organisms[0],
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	for(size_t i = 1; i < p->norganisms; i++){
		p->organisms[i] = nn_ffnet_copy(p->organisms[0]);
	}

	/* Create the starting species */
	p->nspecies = 1;
	p->species = malloc(sizeof(struct neat_species*));
	assert(p->species);

	p->species[0] = neat_species_create(config);

	return p;
}

void neat_destroy(neat_t population)
{
	struct neat_pop *p = population;
	assert(p);

	for(size_t i = 0; i < p->norganisms; i++){
		nn_ffnet_destroy(p->organisms[i]);
	}
	free(p->organisms);
	free(p->organism_fitness);
	free(p->organism_time_alive);

	for(size_t i = 0; i < p->nspecies; i++){
		neat_species_destroy(p->species[i]);
	}
	free(p->species);
	free(p);

	p = NULL;
}

const float *neat_run(neat_t population,
		      size_t organism_id,
		      const float *inputs)
{
	assert(inputs);
	struct neat_pop *p = population;
	assert(p);
	assert(organism_id < p->norganisms);

	return nn_ffnet_run(p->organisms[organism_id], inputs);
}

void neat_epoch(neat_t population)
{
	struct neat_pop *p = population;
	assert(p);

	/* First find the genome with the worst fitness */
	size_t worst_organism;
	bool found_worst = false;

	float worst_fitness = FLT_MAX;
	for(size_t i = 0; i < p->norganisms; i++){
		float fitness = p->organism_fitness[i];
		size_t ticks_alive = p->organism_time_alive[i];
		if(fitness < worst_fitness &&
		   ticks_alive > p->conf.organism_minimum_ticks_alive){
			worst_organism = i;
			worst_fitness = fitness;
			found_worst = true;
		}
	}

	if(!found_worst){
		return;
	}
}

void neat_set_fitness(neat_t population, size_t organism_id, float fitness)
{
	struct neat_pop *p = population;
	assert(p);
	assert(organism_id < p->norganisms);

	p->organism_fitness[organism_id] = fitness;
}

void neat_increase_time_alive(neat_t population, size_t organism_id)
{
	struct neat_pop *p = population;
	assert(p);
	assert(organism_id < p->norganisms);

	p->organism_time_alive[organism_id]++;
}
