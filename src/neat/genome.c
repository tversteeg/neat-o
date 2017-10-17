#include "genome.h"

#include <string.h>
#include <stdint.h>
#include <assert.h>

static void neat_genome_zeroify_innovations(struct neat_genome *genome)
{
	assert(genome);
	assert(genome->net);

	genome->used_weights = 0;
	for(size_t i = 0; i < genome->net->nweights; i++){
		/* Set the innovation to 0 if the weight is 0 */
		int weight_is_set = genome->net->weight[i] != 0;
		genome->innovations[i] *= weight_is_set;
		genome->used_weights += weight_is_set;
	}
}

static size_t neat_genome_allocate_innovations(struct neat_genome *genome)
{
	assert(genome);
	assert(genome->net);

	size_t bytes = sizeof(int) * genome->net->nweights;
	genome->innovations = realloc(genome->innovations, bytes);
	assert(genome->innovations);

	return bytes;
}

static void neat_genome_add_layer(struct neat_genome *genome)
{
	float weight = (float)rand() / (float)(RAND_MAX / 4.0f) - 2.0f;
	genome->net = nn_ffnet_add_hidden_layer(genome->net, weight);

	neat_genome_allocate_innovations(genome);

	neat_genome_zeroify_innovations(genome);
}

static void neat_genome_add_neuron(struct neat_genome *genome, int innovation)
{
	assert(genome);
	assert(genome->net);

	/* Add + 1 to the selection of the layer so a new one can be created */
	size_t layer = rand() % (genome->net->nhidden_layers + 1);
	if(layer >= genome->net->nhidden_layers){
		neat_genome_add_layer(genome);
		return;
	}
}

static void neat_genome_add_link(struct neat_genome *genome, int innovation)
{
	assert(genome);
	assert(genome->net);

	/* Select a random available weight */
	size_t available_weights = genome->net->nweights - genome->used_weights;
	size_t select_weight_offset = rand() % available_weights;

	/* Loop over the available weight to find the randomly selected one */
	size_t selected_index = SIZE_MAX;
	for(size_t i = 0; i < genome->net->nweights; i++){
		if(genome->net->weight[i] == 0.0f && !select_weight_offset--){
			selected_index = i;
			break;
		}
	}

	/* Netwerk is full, return */
	if(selected_index == SIZE_MAX){
		return;
	}

	float weight_val = (float)rand() / (float)(RAND_MAX / 4.0f) - 2.0f;
	genome->net->weight[selected_index] = weight_val;
	genome->innovations[selected_index] = innovation;
	genome->used_weights++;
}

struct neat_genome *neat_genome_create(struct neat_config config,
				       int innovation)
{
	assert(innovation > 0);

	struct neat_genome *genome = calloc(1, sizeof(struct neat_genome));
	assert(genome);

	genome->net = nn_ffnet_create(config.network_inputs,
				      config.network_hidden_nodes,
				      config.network_outputs,
				      1);

	nn_ffnet_set_activations(genome->net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	nn_ffnet_randomize(genome->net);

	genome->innovations = NULL;
	neat_genome_allocate_innovations(genome);
	for(size_t i = 0; i < genome->net->nweights; i++){
		genome->innovations[i] = innovation;
	}

	return genome;
}

struct neat_genome *neat_genome_copy(const struct neat_genome *genome)
{
	struct neat_genome *new = calloc(1, sizeof(struct neat_genome));
	assert(new);

	new->fitness = genome->fitness;

	new->net = nn_ffnet_copy(genome->net);
	assert(new->net);

	size_t bytes = neat_genome_allocate_innovations(new);
	memcpy(new->innovations, genome->innovations, bytes);

	return new;
}

struct neat_genome *neat_genome_reproduce(const struct neat_genome *parent1,
					  const struct neat_genome *parent2)
{
	assert(parent1);
	assert(parent2);

	//TODO implement random crossover
	return neat_genome_copy(parent1);
}

void neat_genome_mutate(struct neat_genome *genome,
			struct neat_config config,
			int innovation)
{
	assert(genome);
	assert(innovation > 0);

	float random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_add_neuron_mutation_probability){
		neat_genome_add_neuron(genome, innovation);
		return;
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_add_link_mutation_probability){
		neat_genome_add_link(genome, innovation);
		return;
	}
}

void neat_genome_destroy(struct neat_genome *genome)
{
	assert(genome);

	nn_ffnet_destroy(genome->net);
	free(genome->innovations);
	free(genome);
}

const float *neat_genome_run(struct neat_genome *genome, const float *inputs)
{
	assert(genome);
	assert(inputs);

	return nn_ffnet_run(genome->net, inputs);
}

void neat_genome_add_random_node(struct neat_genome *genome, int innovation)
{
	assert(genome);
	assert(innovation > 0);

	//TODO implement
}

bool neat_genome_is_compatible(const struct neat_genome *genome,
			       const struct neat_genome *other,
			       float treshold)
{
	//TODO implement by checking the distance between the genomes

	return true;
}

void neat_genome_print_net(const struct neat_genome *genome)
{
	assert(genome);
	assert(genome->net);

	const struct nn_ffnet *n = genome->net;

	float *weight = n->weight;

	printf("\nInputs -> Hiddens: ");
	for(size_t i = 0; i < (n->ninputs + 1) * n->nhiddens; i++){
		printf("%g ", *weight++);
	}
	size_t hidden_weights = (n->nhiddens + 1) * n->nhiddens;
	for(size_t i = 1; i < n->nhidden_layers; i++){
		printf("\n%d Hiddens -> Hiddens: ", (int)i);
		for(size_t j = 0; j < hidden_weights; j++){
			printf("%g ", *weight++);
		}
	}

	printf("\n");
}
