#include "species.h"

#include <stdint.h>
#include <assert.h>

#include "population.h"

static struct neat_genome *neat_genome_at(struct neat_pop *p, size_t index)
{
	struct neat_genome *genome;

	assert(p);
	assert(index < p->ngenomes);

	genome = p->genomes[index];
	assert(genome);

	return genome;
}

struct neat_species *neat_species_create(struct neat_config config)
{
	struct neat_species *species;

	assert(config.population_size > 0);

	species = calloc(1, sizeof(struct neat_species));
	assert(species);

	/* Create all the genomes but don't use them yet, so we don't have
	 * to resize the array when species gets added or removed
	 */
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
		sum_fitness += neat_genome_at(p, species->genomes[i])->fitness;
	}

	species->avg_fitness = sum_fitness / (float)species->ngenomes;

	return species->avg_fitness;
}

size_t neat_species_select_genitor(struct neat_pop *p,
				   struct neat_species *species)
{
	size_t i, best_genome;
	float best_fitness;

	assert(p);
	assert(species);
	assert(species->ngenomes > 0);
	
	/* Return the genome in this species with the highest fitness */
	best_genome = 0;
	best_fitness = 0.0f;
	for(i = 0; i < species->ngenomes; i++){
		float fitness;
	
		fitness = neat_genome_at(p, species->genomes[i])->fitness;
		if(best_fitness < fitness){
			best_fitness = fitness;
			best_genome = i;
		}
	}

	return species->genomes[best_genome];
}

size_t neat_species_select_second_genitor(struct neat_pop *p,
					  struct neat_species *species)
{
	size_t i, best_genome, second_best_genome;
	float best_fitness, second_best_fitness;

	assert(p);
	assert(species);

	/* If there is only one genome in the species there are not that many
	 * options to choose from
	 */
	if(species->ngenomes == 1){
		return species->genomes[0];
	}

	/* TODO choose randomly between one of the best instead of the
	 * second best one, the current algorithm is highly flawed because it
	 * doesn't really select the second best one
	 */
	
	best_genome = second_best_genome = 0;
	best_fitness = second_best_fitness = 0.0f;
	for(i = 0; i < species->ngenomes; i++){
		float fitness;
	
		fitness = neat_genome_at(p, species->genomes[i])->fitness;
		if(best_fitness < fitness){
			second_best_fitness = best_fitness;
			second_best_genome = best_genome;

			best_fitness = fitness;
			best_genome = i;
		}else if(second_best_fitness < fitness){
			second_best_fitness = fitness;
			second_best_genome = i;
		}
	}

	return species->genomes[second_best_genome];
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

	/* Do nothing if the genome is already there */
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
