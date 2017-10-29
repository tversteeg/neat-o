#pragma once

#include <stdio.h>
#include <stdbool.h>

#include <nn.h>

typedef void* neat_t;

struct neat_config{
	/* Neural Networks */
	/* No default */
	size_t network_inputs, network_outputs, network_hidden_nodes;

	/* rtNEAT */
	/* No default */
	size_t population_size;

	/* Default 20 */
	size_t minimum_time_before_replacement;

	/* Species Crossover */
	/* Default 0.2 */
	float species_crossover_probability;
	/* Default 0.05 */
	float interspecies_crossover_probability;
	/* Default 0.3 */
	float mutate_species_crossover_probability;

	/* Genome Mutation */
	/* Default 0.1 */
	float genome_add_neuron_mutation_probability;
	/* Default 0.12 */
	float genome_add_link_mutation_probability;
	/* Default 0.3 */
	float genome_weight_mutation_probability;
	/* Default 0.21 */
	float genome_all_weights_mutation_probability;

	/* Genomes */
	/* Default 100 */
	size_t genome_minimum_ticks_alive;
	/* Default 0.2 */
	float genome_compatibility_treshold;
};

extern const struct neat_config NEAT_DEFAULT_CONFIG;

neat_t neat_create(struct neat_config config);
void neat_destroy(neat_t population);

const float *neat_run(neat_t population, size_t genome_id, const float *inputs);

bool neat_epoch(neat_t population, size_t *worst_genome);

void neat_set_fitness(neat_t population, size_t genome_id, float fitness);

void neat_increase_time_alive(neat_t population, size_t genome_id);

const struct nn_ffnet *neat_get_network(neat_t population, size_t genome_id);
size_t neat_get_species_id(neat_t population, size_t genome_id);

void neat_print_net(neat_t population, size_t genome_id);
