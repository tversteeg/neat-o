#include "genome.h"

#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

static inline float neat_random_two()
{
	return (float)rand() / (float)(RAND_MAX / 4.0f) - 2.0f;
}

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
	assert(bytes > 0);
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

static void neat_genome_add_link(struct neat_genome *genome, int innovation)
{
	assert(genome);
	assert(genome->net);

	/* Select a random available weight */
	size_t available_weights = genome->net->nweights - genome->used_weights;
	size_t select_weight_offset = rand() % available_weights;

	/* Loop over the available weight to find the randomly selected one */
	for(size_t i = 0; i < genome->net->nweights; i++){
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
	assert(genome);
	assert(genome->net);

	struct nn_ffnet *n = genome->net;

	/* Add + 1 to the selection of the layer so a new one can be created */
	size_t layer = rand() % (n->nhidden_layers) + 1;
	if(layer >= n->nhidden_layers){
		neat_genome_add_layer(genome);
		return;
	}

	/* Find the first disconnected layer starting from the selected layer
	 * and set the weight value to a random previous node
	 */
	for(size_t i = n->ninputs + layer * n->nhiddens; i < n->nneurons; i++){
		if(!nn_ffnet_neuron_is_connected(n, i)){
			size_t start = nn_ffnet_get_weight_to_neuron(n, i);
			size_t index = start + (rand() % n->nhiddens);

			n->weight[index] = neat_random_two();
			genome->innovations[index] = innovation;
			genome->used_weights++;
			return;
		}
	}

	/* No available nodes found, just add a new link */
	neat_genome_add_link(genome, innovation);
}

static void neat_genome_mutate_weight(struct neat_genome *genome,
				      int innovation)
{
	assert(genome);
	assert(genome->net);

	if(genome->used_weights == 0){
		return;
	}

	size_t select_weight_offset = rand() % genome->used_weights;

	/* Loop over the available weight to find the randomly selected one */
	for(size_t i = 0; i < genome->net->nweights; i++){
		if(genome->net->weight[i] != 0.0f && !select_weight_offset--){
			genome->net->weight[i] = neat_random_two();
			return;
		}
	}
}

static void neat_genome_mutate_all_weights(struct neat_genome *genome,
					   int innovation)
{
	assert(genome);
	assert(genome->net);

	/* Loop over the available weight to find the randomly selected one */
	for(size_t i = 0; i < genome->net->nweights; i++){
		float *weight = genome->net->weight + i;
		if(*weight != 0.0f){
			*weight = neat_random_two();
			return;
		}
	}
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
				 NN_ACTIVATION_PASSTHROUGH,
				 NN_ACTIVATION_SIGMOID);

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

	new->innovations = NULL;
	size_t bytes = neat_genome_allocate_innovations(new);
	memcpy(new->innovations, genome->innovations, bytes);
	assert(new->innovations);

	return new;
}

struct neat_genome *neat_genome_reproduce(const struct neat_genome *parent1,
					  const struct neat_genome *parent2)
{
	assert(parent1);
	assert(parent2);

	/* Take the biggest parent as the base */
	if(parent2->net->nweights > parent1->net->nweights){
		const struct neat_genome *tmp = parent1;
		parent1 = parent2;
		parent2 = tmp;
	}

	struct neat_genome *child = neat_genome_copy(parent1);

	//TODO fix enabling
	for(size_t i = 0; i < parent2->net->nweights; i++){
		int in1 = parent1->innovations[i];
		int in2 = parent2->innovations[i];
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

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_weight_mutation_probability){
		neat_genome_mutate_weight(genome, innovation);
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_all_weights_mutation_probability){
		neat_genome_mutate_all_weights(genome, innovation);
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
	assert(genome);
	assert(other);
	assert(genome->net);
	assert(other->net);
	assert(genome->innovations);
	assert(other->innovations);

	size_t weights1 = genome->net->nweights;
	size_t weights2 = other->net->nweights;
	size_t min_weights;
	size_t max_weights;
	if(weights1 < weights2){
		min_weights = weights1;
		max_weights = weights2;
	}else{
		min_weights = weights2;
		max_weights = weights1;
	}

	size_t excess = max_weights - min_weights;
	size_t disjoint = 0;
	size_t matching = 0;
	float weight_sum = 0.0;

	for(size_t i = 0; i < min_weights; i++){
		if(genome->innovations[i] == other->innovations[i]){
			float weight1 = genome->net->weight[i];
			float weight2 = other->net->weight[i];
			weight_sum += fabs(weight1 - weight2);
			matching++;
		}else{
			disjoint++;
		}
	}

	float distance = 1.0 * excess / (float)max_weights;
	distance += 1.5 * disjoint / (float)max_weights;
	distance += 0.4 * weight_sum / (float)matching;

	return distance < treshold;
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
			if((int)(*weight * 10.0) != 0.0){
				printf("%d:%.1f ",
				       (int)(j / n->nhiddens),
				       *weight);
			}
			
			weight++;
		}
	}

	printf("\n");
}
