#pragma once

#include <stdio.h>
#include <stdbool.h>

#include <nn.h>

typedef void* neat_t;

struct neat_config{
	/* Neural Networks */
	size_t network_inputs, network_outputs, network_hidden_nodes;

	/* rtNEAT */
	size_t population_size;
	size_t minimum_time_before_replacement;

	/* Species */
	size_t species_stagnation_treshold;
	size_t species_stagnations_allowed;

	/* Species Crossover */
	float species_crossover_probability;
	float interspecies_crossover_probability;

	/* Genome Mutation */
	float genome_add_neuron_mutation_probability;
	float genome_add_link_mutation_probability;
	float genome_change_activation_probability;
	float genome_weight_mutation_probability;
	float genome_all_weights_mutation_probability;

	/* Genomes */
	size_t genome_minimum_ticks_alive;
	float genome_compatibility_treshold;
	
	enum nn_activation genome_default_hidden_activation;
	enum nn_activation genome_default_output_activation;
};

struct neat_config neat_get_default_config(void);

neat_t neat_create(struct neat_config config);
void neat_destroy(neat_t population);

const float *neat_run(neat_t population, size_t genome_id, const float *inputs);

bool neat_epoch(neat_t population, size_t *worst_genome);

void neat_set_fitness(neat_t population, size_t genome_id, float fitness);

void neat_increase_time_alive(neat_t population, size_t genome_id);

const struct nn_ffnet *neat_get_network(neat_t population, size_t genome_id);
size_t neat_get_species_id(neat_t population, size_t genome_id);

size_t neat_get_num_species(neat_t population);
size_t neat_get_num_genomes_in_species(neat_t population, size_t species_id);
float neat_get_average_fitness_of_species(neat_t population, size_t species_id);
bool neat_get_species_is_alive(neat_t population, size_t species_id);

void neat_print_net(neat_t population, size_t genome_id);
