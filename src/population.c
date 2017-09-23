#include "population.h"

#include <stdlib.h>

#include "network.h"
#include "species.h"

static void neat_create_new_species(struct neat_pop *p)
{
	size_t bytes = sizeof(struct neat_species) * (p->nspecies + 1);
	p->species = realloc(p->species, bytes);

	struct neat_species *s = p->species + p->nspecies;

	*s = neat_species_create(p->conf, p->nspecies, &p->initial_genome);

	p->nspecies++;
}

neat_pop_t neat_population_create(struct neat_config config)
{
	struct neat_pop *p = malloc(sizeof(struct neat_pop));

	p->conf = config;
	p->initial_genome = neat_ffnet_create(config);

	p->species = malloc(sizeof(struct neat_species));
	*p->species = neat_species_create(p->conf, 0, &p->initial_genome);
	p->nspecies = 1;

	return p;
}

neat_genome_t neat_run(neat_pop_t population,
		       const double *inputs,
		       double(*fitness_func)(double *outputs),
		       int generations)
{
	struct neat_pop *p = population;

	for(int gen = 0; gen < generations; gen++){
		for(int i = 0; i < p->nspecies; i++){
			struct neat_species *s = p->species + i;

			double avg_fitness;
			bool solved = neat_species_run(s, inputs, fitness_func,
						       &avg_fitness);
			neat_species_evolve(s);
		}
	}

	return NULL;
}

bool neat_is_solved(neat_pop_t population)
{
	return ((struct neat_pop*)population)->solved;
}
