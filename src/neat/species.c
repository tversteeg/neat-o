#include "species.h"

#include <assert.h>

struct neat_species *neat_species_create(struct neat_config config,
					 struct neat_genome *base_genome)
{
	assert(config.population_size > 0);

	struct neat_species *species = calloc(1, sizeof(struct neat_species));
	assert(species);

	/* Create all the genomes but don't use them yet */
	species->genomes = calloc(config.population_size,
				  sizeof(struct neat_genome*));
	assert(species->genomes);

	if(base_genome == NULL){
		species->ngenomes = 0;
	}else{
		species->ngenomes = config.population_size;
		for(size_t i = 0; i < species->ngenomes; i++){
			species->genomes[i] = base_genome;
		}
	}

	return species;
}

void neat_species_destroy(struct neat_species *species)
{
	assert(species);
	assert(species->genomes);

	free(species->genomes);
	free(species);
}

float neat_species_get_adjusted_fitness(struct neat_species *species,
					float fitness)
{
	assert(species);
	assert(species->ngenomes > 0);

	return fitness / (float)species->ngenomes;
}

float neat_species_get_average_fitness(struct neat_species *species)
{
	assert(species);

	float sum_fitness = 0.0f;
	for(size_t i = 0; i < species->ngenomes; i++){
		/* We can get the adjusted fitness for every node by dividing
		 * them all but it's better to do one divide at the end
		 */
		sum_fitness += species->genomes[i]->fitness;
	}

	return sum_fitness / (float)(species->ngenomes * 2);
}

struct neat_genome *neat_species_select_genitor(struct neat_species *species)
{
	assert(species);
	assert(species->ngenomes > 0);

	return species->genomes[rand() % species->ngenomes];
}

struct neat_genome *neat_species_get_representant(struct neat_species *species)
{
	assert(species);
	assert(species->ngenomes > 0);

	//TODO track the representant, for now just return a random genome

	return species->genomes[rand() % species->ngenomes];
}

void neat_species_add_genome(struct neat_species *species,
			     struct neat_genome *genome)
{
	assert(species);
	assert(genome);

	species->genomes[species->ngenomes] = genome;
	species->ngenomes++;
}

void neat_species_remove_genome(struct neat_species *species,
				struct neat_genome *genome)
{
	assert(species);
	assert(genome);

	for(size_t i = 0; i < species->ngenomes; i++){
		if(species->genomes[i] != genome){
			continue;
		}

		/* Put the last genome on this position
		 * (this will do nothing if it already is the last one)
		 */

		printf("%p\n", (void*)species->genomes[i]);
		species->genomes[i] = species->genomes[--species->ngenomes];
		species->genomes[species->ngenomes] = NULL;
		printf("%p\n", (void*)species->genomes[i]);

		return;
	}
}

bool neat_species_contains_genome(struct neat_species *species,
				  struct neat_genome *genome)
{
	assert(species);
	assert(genome);

	for(size_t i = 0; i < species->ngenomes; i++){
		if(species->genomes[i] == genome){
			return true;
		}
	}

	return false;
}
