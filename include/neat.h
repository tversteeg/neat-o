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
	size_t network_hidden_nodes, network_hidden_layers;
};

neat_pop_t neat_population_create(struct neat_config config);
void neat_population_destroy(neat_pop_t population);
