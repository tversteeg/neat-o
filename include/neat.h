#pragma once

#include <stdio.h>
#include <stdbool.h>

#include <nn.h>

typedef void* neat_t;

struct neat_config{
	/* NEAT */
	size_t population_size;
	bool reset_on_extinction;

	/* Species */
	double species_crossover_probability;

	/* Organisms */
	size_t genome_minimum_ticks_alive;

	/* Neural Networks */
	size_t network_inputs, network_outputs;
	size_t network_hidden_nodes, network_hidden_layers;
};

neat_t neat_create(struct neat_config config);
void neat_destroy(neat_t population);

const float *neat_run(neat_t population, size_t genome_id, const float *inputs);

void neat_epoch(neat_t population);

void neat_set_fitness(neat_t population, size_t genome_id, float fitness);

void neat_increase_time_alive(neat_t population, size_t genome_id);
