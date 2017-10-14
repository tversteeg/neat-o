#pragma once

#include <neat.h>

#include "genome.h"

struct neat_species{
	bool active;

	struct neat_genome **genomes;
	size_t ngenomes;
};

struct neat_species *neat_species_create(struct neat_config config,
					 struct neat_genome *base_genome);
void neat_species_destroy(struct neat_species *species);

float neat_species_get_adjusted_fitness(struct neat_species *species,
					float fitness);
float neat_species_get_average_fitness(struct neat_species *species);

struct neat_genome *neat_species_select_genitor(struct neat_species *species);

struct neat_genome *neat_species_get_representant(struct neat_species *species);

void neat_species_add_genome(struct neat_species *species,
			     struct neat_genome *genome);
void neat_species_remove_genome(struct neat_species *species,
				struct neat_genome *genome);
bool neat_species_contains_genome(struct neat_species *species,
				  struct neat_genome *genome);

