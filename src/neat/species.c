#include "species.h"

#include <stdint.h>
#include <assert.h>

#include "population.h"

static struct neat_genome *neat_genome_at(struct neat_pop *p, size_t index)
{
	assert(p);
	assert(index < p->ngenomes);

	return p->genomes[index];
}

struct neat_species *neat_species_create(struct neat_config config)
{
	struct neat_species *species;

	assert(config.population_size > 0);

	species = calloc(1, sizeof(struct neat_species));
	assert(species);

	/* Create all the genomes but don't use them yet */
	species->genomes = calloc(config.population_size, sizeof(size_t));
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

float neat_species_get_adjusted_fitness(struct neat_species *species,
					float fitness)
{
	assert(species);
	assert(species->ngenomes > 0);

	return fitness / (float)species->ngenomes;
}

float neat_species_update_average_fitness(struct neat_pop *p,
					  struct neat_species *species)
{
	float sum_fitness;
	size_t i;

	assert(p);
	assert(species);

	sum_fitness = 0.0f;
	for(i = 0; i < species->ngenomes; i++){
		sum_fitness += neat_genome_at(p, i)->fitness;
	}

	species->avg_fitness = sum_fitness / (float)species->ngenomes;

	return species->avg_fitness;
}

size_t neat_species_select_genitor(struct neat_species *species)
{
	size_t index;

	assert(species);
	assert(species->ngenomes > 0);

	index = rand() % species->ngenomes;

	return species->genomes[index];
}

size_t neat_species_get_representant(struct neat_species *species)
{
	assert(species);
	assert(species->ngenomes > 0);

	/* The representant will be the first genome assigned to this species */

	return species->genomes[0];
}

void neat_species_add_genome(struct neat_species *species,
			     size_t genome_id)
{
	size_t i;

	assert(species);

	/* Check if the genome is already there */
	for(i = 0; i < species->ngenomes; i++){
		if(species->genomes[i] == genome_id){
			return;
		}
	}

	species->genomes[species->ngenomes] = genome_id;
	species->ngenomes++;
}

bool neat_species_remove_genome_if_exists(struct neat_species *species,
					  size_t genome_id)
{
	size_t i;

	assert(species);

	for(i = 0; i < species->ngenomes; i++){
		if(species->genomes[i] != genome_id){
			continue;
		}

		/* Put the last genome on this position
		 * (this will do nothing if it already is the last one)
		 */
		species->genomes[i] = species->genomes[--species->ngenomes];
		species->genomes[species->ngenomes] = SIZE_MAX;

		return true;
	}

	return false;
}

bool neat_species_contains_genome(struct neat_species *species,
				  size_t genome_id)
{
	size_t i;

	assert(species);

	for(i = 0; i < species->ngenomes; i++){
		if(species->genomes[i] == genome_id){
			return true;
		}
	}

	return false;
}
