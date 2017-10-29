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
} NEAT_DEFAULT_CONFIG = {
	/* These variables need to be set because they are specific for each
	 * implementation
	 */
	0, 0, 0, 0,

	/* Minimum time before replacement */
	20,

	/* Species crossover probability */
	0.2,
	/* Interspecies crossover probability */
	0.05,
	/* Mutate species crossover probability */
	0.3,
	
	/* Genome add neuron mutation probability */
	0.1,
	/* Genome add link mutation probability */
	0.12,
	/* Genome weight mutation probability */
	0.3,
	/* Genome mutate all weights probability */
	0.21,

	/* Genome minimum ticks alive */
	100,
	/* Genome compatibility treshold */
	0.2
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
