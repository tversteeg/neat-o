#pragma once

#include <stdio.h>
#include <stdbool.h>

#include <nn.h>

typedef void* neat_t;

struct neat_config{
	/* Neural Networks */
	size_t network_inputs, network_outputs;
	size_t network_hidden_nodes;

	/* rtNEAT */
	size_t population_size;
	bool reset_on_extinction;
	bool speciate;

	size_t minimum_time_before_replacement;

	/* Species Crossover */
	float species_crossover_probability;
	float interspecies_crossover_probability;
	float mutate_species_crossover_probability;
	/* Genome Mutation */
	float genome_add_neuron_mutation_probability;
	float genome_add_link_mutation_probability;
	float genome_weight_mutation_probability;
	float genome_all_weights_mutation_probability;

	/* Genomes */
	size_t genome_minimum_ticks_alive;
	float genome_compatibility_treshold;
};

neat_t neat_create(struct neat_config config);
void neat_destroy(neat_t population);

const float *neat_run(neat_t population, size_t genome_id, const float *inputs);

bool neat_epoch(neat_t population, size_t *worst_genome);

void neat_set_fitness(neat_t population, size_t genome_id, float fitness);

void neat_increase_time_alive(neat_t population, size_t genome_id);

const struct nn_ffnet *neat_get_network(neat_t population, size_t genome_id);
size_t neat_get_species_id(neat_t population, size_t genome_id);

void neat_print_net(neat_t population, size_t genome_id);
