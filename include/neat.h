#pragma once

#include <stdio.h>
#include <stdbool.h>

typedef void* neat_pop_t;
typedef void* neat_genome_t;
typedef void* neat_ffnet_t;

enum neat_fitness_criterion{
	NEAT_FITNESS_CRITERION_MAX,
	NEAT_FITNESS_CRITERION_MIN,
	NEAT_FITNESS_CRITERION_MEAN
};

enum neat_activation_option{
	NEAT_ACTIVATION_SIGMOID
};

struct neat_config{
	/* NEAT */
	enum neat_fitness_criterion fitness_criterion;
	double fitness_treshold;
	int population_size;
	bool reset_on_extinction;

	int input_genome_topo, output_genome_topo;

	/* Default Genome */
	enum neat_activation_option activation_default, activation_option;
};

neat_pop_t neat_population_create(struct neat_config config);
neat_genome_t neat_run(neat_pop_t population,
		       double(*fitness_func)(double *outputs),
		       int generations);

bool neat_is_solved(neat_pop_t population);
