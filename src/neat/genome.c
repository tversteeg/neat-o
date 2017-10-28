#include "genome.h"

#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

static float neat_random_two()
{
	return (float)rand() / (float)(RAND_MAX / 4.0f) - 2.0f;
}

static void neat_genome_zeroify_innovations(struct neat_genome *genome)
{
	size_t i;

	assert(genome);
	assert(genome->net);

	genome->used_weights = 0;
	for(i = 0; i < genome->net->nweights; i++){
		int weight_is_set;

		/* Set the innovation to 0 if the weight is 0 */
		weight_is_set = genome->net->weight[i] != 0;
		genome->innovations[i] *= weight_is_set;
		genome->used_weights += weight_is_set;
	}
}

static size_t neat_genome_allocate_innovations(struct neat_genome *genome,
					       int innovation)
{
	int *innov;
	size_t bytes, diff, i;

	assert(genome);
	assert(genome->net);

	bytes = sizeof(int) * genome->net->nweights;
	assert(bytes > 0);
	genome->innovations = realloc(genome->innovations, bytes);
	assert(genome->innovations);

	diff = genome->net->nweights - genome->ninnovations;
	innov = genome->innovations + genome->ninnovations;
	for(i = 0; i < diff; i++){
		*innov++ = innovation;
	}

	genome->ninnovations = genome->net->nweights;

	return bytes;
}

static void neat_genome_add_layer(struct neat_genome *genome, int innovation)
{
	float weight;

	weight = (float)rand() / (float)(RAND_MAX / 4.0f) - 2.0f;
	genome->net = nn_ffnet_add_hidden_layer(genome->net, weight);

	neat_genome_allocate_innovations(genome, innovation);
	neat_genome_zeroify_innovations(genome);
}

static void neat_genome_add_link(struct neat_genome *genome, int innovation)
{
	size_t available_weights, select_weight_offset, i;

	assert(genome);
	assert(genome->net);

	/* Select a random available weight */
	available_weights = genome->net->nweights - genome->used_weights;
	select_weight_offset = rand() % available_weights;

	/* Loop over the available weight to find the randomly selected one */
	for(i = 0; i < genome->net->nweights; i++){
		if(genome->net->weight[i] == 0.0f && !select_weight_offset--){
			genome->net->weight[i] = neat_random_two();
			genome->innovations[i] = innovation;
			genome->used_weights++;
			return;
		}
	}
}

static void neat_genome_add_neuron(struct neat_genome *genome, int innovation)
{
	struct nn_ffnet *n;
	size_t layer, i;

	assert(genome);
	assert(genome->net);

	n = genome->net;

	/* Add + 1 to the selection of the layer so a new one can be created */
	layer = rand() % (n->nhidden_layers) + 1;
	if(layer >= n->nhidden_layers){
		neat_genome_add_layer(genome, innovation);
		return;
	}

	/* Find the first disconnected layer starting from the selected layer
	 * and set the weight value to a random previous node
	 */
	for(i = n->ninputs + layer * n->nhiddens; i < n->nneurons; i++){
		if(!nn_ffnet_neuron_is_connected(n, i)){
			size_t start, index;

			start = nn_ffnet_get_weight_to_neuron(n, i);
			index = start + (rand() % n->nhiddens);

			n->weight[index] = neat_random_two();
			genome->innovations[index] = innovation;
			genome->used_weights++;
			return;
		}
	}

	/* No available nodes found, just add a new link */
	neat_genome_add_link(genome, innovation);
}

static void neat_genome_mutate_weight(struct neat_genome *genome)
{
	size_t select_weight_offset, i;

	assert(genome);
	assert(genome->net);

	if(genome->used_weights == 0){
		return;
	}

	select_weight_offset = rand() % genome->used_weights;

	/* Loop over the available weight to find the randomly selected one */
	for(i = 0; i < genome->net->nweights; i++){
		if(genome->net->weight[i] != 0.0f && !select_weight_offset--){
			genome->net->weight[i] = neat_random_two();
			return;
		}
	}
}

static void neat_genome_mutate_all_weights(struct neat_genome *genome)
{
	size_t i;

	assert(genome);
	assert(genome->net);

	/* Loop over the available weight to find the randomly selected one */
	for(i = 0; i < genome->net->nweights; i++){
		float *weight;

		weight = genome->net->weight + i;
		if(*weight != 0.0f){
			*weight = neat_random_two();
			return;
		}
	}
}

struct neat_genome *neat_genome_create(struct neat_config config,
				       int innovation)
{
	struct neat_genome *genome;
	size_t i;

	assert(innovation > 0);
	assert(config.network_inputs > 0);
	assert(config.network_hidden_nodes > 0);
	assert(config.network_outputs > 0);

	genome = calloc(1, sizeof(struct neat_genome));
	assert(genome);

	genome->net = nn_ffnet_create(config.network_inputs,
				      config.network_hidden_nodes,
				      config.network_outputs,
				      0);
	assert(genome->net);

	nn_ffnet_set_activations(genome->net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_SIGMOID);

	nn_ffnet_randomize(genome->net);

	neat_genome_allocate_innovations(genome, innovation);
	for(i = 0; i < genome->net->nweights; i++){
		genome->innovations[i] = innovation;
	}
	neat_genome_zeroify_innovations(genome);

	return genome;
}

struct neat_genome *neat_genome_copy(const struct neat_genome *genome)
{
	struct neat_genome *new;
	size_t bytes;

	new = calloc(1, sizeof(struct neat_genome));
	assert(new);

	new->fitness = genome->fitness;

	new->net = nn_ffnet_copy(genome->net);
	assert(new->net);

	new->innovations = NULL;
	bytes = neat_genome_allocate_innovations(new, 1);
	memcpy(new->innovations, genome->innovations, bytes);
	assert(new->innovations);

	return new;
}

struct neat_genome *neat_genome_reproduce(const struct neat_genome *parent1,
					  const struct neat_genome *parent2)
{
	struct neat_genome *child;
	size_t i;

	assert(parent1);
	assert(parent2);

	/* Take the biggest parent as the base */
	if(parent2->net->nweights > parent1->net->nweights){
		const struct neat_genome *tmp;

		tmp = parent1;
		parent1 = parent2;
		parent2 = tmp;
	}

	child = neat_genome_copy(parent1);

	/* TODO fix enabling */
	for(i = 0; i < parent2->net->nweights; i++){
		int in1, in2;

		in1 = parent1->innovations[i];
		in2 = parent2->innovations[i];
		if(in1 == in2){
			if(parent1->fitness < parent2->fitness){
				child->net->weight[i] = parent2->net->weight[i];
			}
			continue;
		}
		if(in1 == 0 && in2 != 0){
			child->net->weight[i] = parent2->net->weight[i];
			child->innovations[i] = parent2->innovations[i];
			child->used_weights++;
		}
	}

	return child;
}

void neat_genome_mutate(struct neat_genome *genome,
			struct neat_config config,
			int innovation)
{
	float random;

	assert(genome);
	assert(innovation > 0);

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_add_neuron_mutation_probability){
		neat_genome_add_neuron(genome, innovation);
		return;
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_add_link_mutation_probability){
		neat_genome_add_link(genome, innovation);
		return;
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_weight_mutation_probability){
		neat_genome_mutate_weight(genome);
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_all_weights_mutation_probability){
		neat_genome_mutate_all_weights(genome);
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

	/* TODO implement */
}

bool neat_genome_is_compatible(const struct neat_genome *genome,
			       const struct neat_genome *other,
			       float treshold)
{
	size_t i, excess, disjoint, matching;
	size_t weights1, weights2, min_weights, max_weights;
	float weight_sum, distance;

	assert(genome);
	assert(other);
	assert(genome->net);
	assert(other->net);
	assert(genome->innovations);
	assert(other->innovations);

	weights1 = genome->ninnovations;
	weights2 = other->ninnovations;
	if(weights1 < weights2){
		min_weights = weights1;
		max_weights = weights2;
	}else{
		min_weights = weights2;
		max_weights = weights1;
	}

	excess = max_weights - min_weights;
	disjoint = 0;
	matching = 0;
	weight_sum = 0.0;

	for(i = 0; i < min_weights; i++){
		if(genome->innovations[i] == other->innovations[i]){
			float weight1, weight2;

			weight1 = genome->net->weight[i];
			weight2 = other->net->weight[i];
			weight_sum += fabs(weight1 - weight2);
			matching++;
		}else{
			disjoint++;
		}
	}

	distance = 1.0 * excess / (float)max_weights;
	distance += 1.5 * disjoint / (float)max_weights;
	distance += 0.4 * weight_sum / (float)matching;

	return distance < treshold;
}

void neat_genome_print_net(const struct neat_genome *genome)
{
	const struct nn_ffnet *n;
	float *weight;
	size_t i;

	assert(genome);
	assert(genome->net);

	n = genome->net;

	weight = n->weight;

	printf("\nInputs -> Hiddens: ");
	for(i = 0; i < n->nhiddens; i++){
		size_t j;

		if(i != 0){
			printf(": ");
		}
		for(j = 0; j < n->ninputs + 1; j++){
			printf("%g ", *weight++);
		}
	}
	for(i = 0; i < n->nhidden_layers - 1; i++){
		size_t j;

		printf("\nHiddens -> Hiddens: ");
		for(j = 0; j < n->nhiddens; j++){
			size_t k;

			if(j != 0){
				printf(": ");
			}
			for(k = 0; k < n->nhiddens + 1; k++){
				printf("%g ", *weight++);
			}
		}
	}

	printf("\nHiddens -> Outputs: ");
	for(i = 0; i < n->noutputs; i++){
		size_t j;

		if(i != 0){
			printf(": ");
		}
		for(j = 0; j < n->nhiddens + 1; j++){
			printf("%g ", *weight++);
		}
	}
	printf("\n");
}
