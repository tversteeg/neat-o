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
	size_t species_ticks_before_reassignment;

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

/* Create a new population */
neat_t neat_create(struct neat_config config);
/* Destroy the population, this frees all the memory allocated by it */
void neat_destroy(neat_t population);

/* Run the neural network
 * genome_id	id of the genome where the network resides
 * inputs	array of floats to use as the inputs, the amount is defined by
 * 		the "network_inputs" field in the config
 *
 * return an array of outputs as run through the network. the amount is defined
 * by the "network_outputs" field in the config
 */
const float *neat_run(neat_t population, size_t genome_id, const float *inputs);

/* Set the fitness for a run genome
 * genome_id	id of the genome
 * fitness	number between 0.0 and 1.0 that determines how close to the goal
 *		the network was after neat_run
 */
void neat_set_fitness(neat_t population, size_t genome_id, float fitness);

/* Increase the time a genome is alive, should be used every tick
 * genome_id	id of the genome to increase the time alive of
 */
void neat_increase_time_alive(neat_t population, size_t genome_id);

/* Update the whole population, this will check if any genomes died and
 * reproduces them if so
 * worst_genome	pointer to the genome that will be replaced, NULL if nothing is
 * 		replaced
 *
 * return a boolean determining if a genome got replaced
 */
bool neat_epoch(neat_t population, size_t *worst_genome);

const struct nn_ffnet *neat_get_network(neat_t population, size_t genome_id);
size_t neat_get_species_id(neat_t population, size_t genome_id);

size_t neat_get_num_species(neat_t population);
size_t neat_get_num_genomes_in_species(neat_t population, size_t species_id);
float neat_get_average_fitness_of_species(neat_t population, size_t species_id);
bool neat_get_species_is_alive(neat_t population, size_t species_id);

void neat_print_net(neat_t population, size_t genome_id);
