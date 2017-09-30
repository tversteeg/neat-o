#pragma once

#include <neat.h>
#include <nn.h>

struct neat_species{
	int id, population, generation;
	bool active;

	int generation_with_max_fitness, times_stagnated;
	double max_avg_fitness;

	struct neat_ffnet *genome_representative;
	struct neat_ffnet *genomes;
};

struct neat_species neat_species_create(struct neat_config config, int id,
					struct neat_ffnet *genome);
void neat_species_destroy(struct neat_species *species);

bool neat_species_run(struct neat_species *species,
		      double(*fitness_func)(neat_ffnet_t network),
		      double *avg_fitness);

void neat_species_evolve(struct neat_species *species);