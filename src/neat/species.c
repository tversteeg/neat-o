#include "species.h"

#include <assert.h>

struct neat_species *neat_species_create(struct neat_config config,
					 struct nn_ffnet *base_genome)
{
	assert(base_genome);
	assert(config.population_size > 0);

	struct neat_species *species = calloc(sizeof(struct neat_species), 1);
	assert(species);

	/* Create population_size copies of the base genome */
	species->ngenomes = config.population_size;
	species->genomes = malloc(sizeof(struct nn_ffnet*) * species->ngenomes);
	assert(species->genomes);

	for(size_t i = 0; i < species->ngenomes; i++){
		species->genomes[i] = nn_ffnet_copy(base_genome);
	}

	return species;
}

void neat_species_destroy(struct neat_species *species)
{
	assert(species);
	assert(species->genomes);

	for(size_t i = 0; i < species->ngenomes; i++){
		nn_ffnet_destroy(species->genomes[i]);
	}
	free(species->genomes);
	free(species);
}
