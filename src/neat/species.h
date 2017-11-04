#pragma once

#include <neat.h>

#include "genome.h"

/* Forward declare against cyclic dependency */
struct neat_pop;

struct neat_species{
	bool active;

	float avg_fitness, max_avg_fitness;
	size_t generation, generation_with_max_fitness;
	size_t times_stagnated;

	size_t *genomes;
	size_t ngenomes;
};

struct neat_species *neat_species_create(struct neat_config config);
void neat_species_destroy(struct neat_species *species);

float neat_species_get_adjusted_fitness(struct neat_species *species,
					float fitness);
float neat_species_update_average_fitness(struct neat_pop *p,
					  struct neat_species *species);

/* Disable the species based on the number of stagnations and on weakness of
 * the species (amount of genomes)
 */
bool neat_species_cull(struct neat_pop *p, struct neat_species *species);

size_t neat_species_select_genitor(struct neat_pop *p,
				   struct neat_species *species);

size_t neat_species_select_second_genitor(struct neat_pop *p,
					  struct neat_species *species);

size_t neat_species_get_representant(struct neat_species *species);

void neat_species_add_genome(struct neat_species *species,
			     size_t genome_id);
bool neat_species_remove_genome_if_exists(struct neat_species *species,
					  size_t genome_id);
bool neat_species_contains_genome(struct neat_species *species,
				  size_t genome_id);

