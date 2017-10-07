#pragma once

#include <stdio.h>
#include <stdbool.h>

#include <nn.h>

typedef void* neat_pop_t;

struct neat_config{
	/* NEAT */
	size_t population_size;
	bool reset_on_extinction;

	/* Species */
	double species_crossover_probability;

	/* Organisms */
	size_t organism_minimum_ticks_alive;

	/* Neural Networks */
	size_t network_inputs, network_outputs;
	size_t network_max
} neat_default_config = {
	.population_size = 20,
	.reset_on_extinction = true,
	.crossover_probability = 0.15,
	.minimum_ticks_alive = 100
};

neat_pop_t neat_population_create(struct neat_config config);
void neat_population_destroy(neat_pop_t population);
