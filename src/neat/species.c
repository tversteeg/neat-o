#include "species.h"

#include <assert.h>

struct neat_species *neat_species_create(struct neat_config config)
{
	assert(config.population_size > 0);

	struct neat_species *species = calloc(sizeof(struct neat_species), 1);
	assert(species);

	/* Create all the genomes but don't use them yet */
	species->ngenomes = config.population_size;
	species->genomes = calloc(species->ngenomes, sizeof(struct nn_ffnet*));
	assert(species->genomes);

	return species;
}

void neat_species_destroy(struct neat_species *species)
{
	assert(species);
	assert(species->genomes);

	free(species->genomes);
	free(species);
}
