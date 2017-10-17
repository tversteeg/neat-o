#pragma once

#include <stdio.h>
#include <stdbool.h>

#include <nn.h>

typedef void* neat_t;

struct neat_config{
	/* NEAT */
	size_t population_size;
	bool reset_on_extinction;

	/* Species Crossover */
	float species_crossover_probability;
	float interspecies_crossover_probability;
	float mutate_species_crossover_probability;
	/* Genome Mutation */
	float genome_add_neuron_mutation_probability;
	float genome_add_link_mutation_probability;

	/* Genomes */
	size_t genome_minimum_ticks_alive;
	float genome_compatibility_treshold;

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

void neat_print_net(neat_t population, size_t genome_id);
