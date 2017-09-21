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
		    void(*fitness_func)(neat_genome_t *genomes),
		    int generations);
bool neat_is_solved(neat_pop_t population);

neat_ffnet_t neat_ffnet_create(neat_pop_t population);
neat_ffnet_t neat_ffnet_activate(neat_ffnet_t net, double *inputs, int ninputs);

int neat_ffnet_get_outputs(neat_ffnet_t net, double **outputs);
double neat_ffnet_get_output_at_index(neat_ffnet_t net, int index);

void neat_genome_decrease_fitness(neat_genome_t genome, double fitness);
