#include "genome.h"

#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

static float neat_random_two(void)
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
	size_t available, select_weight_offset, i;

	assert(genome);
	assert(genome->net);

	/* TODO change to finding new empty links with passthrough and
	 * connecting those
	 */

	/* Select a random available weight */
	available = genome->net->nweights - genome->used_weights;
	/* Do nothing if there are no more available weights */
	if(available == 0){
		return;
	}

	select_weight_offset = rand() % available;

	/* Loop over the available weight to find the randomly selected one */
	for(i = 0; i < genome->net->nweights; i++){
		if(genome->net->weight[i] != 0.0f){
			continue;
		}

		if(!select_weight_offset--){
			genome->net->weight[i] = neat_random_two();
			genome->innovations[i] = innovation;
			genome->used_weights++;
			return;
		}
	}
}

static void neat_genome_add_neuron(struct neat_genome *genome,
				   int innovation,
				   enum nn_activation default_hidden,
				   enum nn_activation default_output)
{
	struct nn_ffnet *n;
	size_t layer, i;

	assert(genome);
	assert(genome->net);

	n = genome->net;

	/* Add + 1 to the selection of the layer so a new one can be created */
	if(n->nhidden_layers == 0){
		layer = 0;
		neat_genome_add_layer(genome, innovation);
		/* Update the pointer */
		n = genome->net;
	}else{
		layer = rand() % (n->nhidden_layers + 1);
		if(layer >= n->nhidden_layers){
			neat_genome_add_layer(genome, innovation);
			return;
		}
	}

	/* Find the first disconnected layer starting from the selected layer
	 * and set the weight value to a random previous node
	 */
	for(i = n->ninputs + layer * n->nhiddens; i < n->nneurons; i++){
		if(!nn_ffnet_neuron_is_connected(n, i)){
			size_t start, weight_index;

			start = nn_ffnet_get_weight_to_neuron(n, i);
			weight_index = start + (rand() % n->nhiddens);

			n->weight[weight_index] = neat_random_two();
			genome->innovations[weight_index] = innovation;
			genome->used_weights++;
			/* Set the output activation if it's the last layer */
			if(layer == n->nhidden_layers){
				n->activation[i - n->ninputs] = default_output;
			}else{
				n->activation[i - n->ninputs] = default_hidden;
			}
			return;
		}
	}

	/* No available nodes found, just add a new link */
	neat_genome_add_link(genome, innovation);
}

static void neat_genome_mutate_activation(struct neat_genome *genome,
					  int innovation)
{
	size_t random_activ;
	char new_activation;

	/* TODO apply innovation */
	(void)innovation;

	assert(genome);
	assert(genome->net);

	random_activ = rand() % genome->net->nactivations;

	new_activation = rand() % _NN_ACTIVATION_COUNT;
	if(genome->net->activation[random_activ] == new_activation){
		new_activation = (new_activation + 1) % _NN_ACTIVATION_COUNT;
	}

	genome->net->activation[random_activ] = new_activation;
}

static void neat_genome_mutate_weight(struct neat_genome *genome,
				      int innovation)
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
			genome->innovations[i] = innovation;
			return;
		}
	}
}

static void neat_genome_mutate_all_weights(struct neat_genome *genome,
					   int innovation)
{
	size_t i;

	assert(genome);
	assert(genome->net);

	for(i = 0; i < genome->net->nweights; i++){
		float *weight;

		weight = genome->net->weight + i;
		if(*weight != 0.0f){
			*weight = neat_random_two();
			genome->innovations[i] = innovation;
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
				 config.genome_default_hidden_activation,
				 config.genome_default_output_activation);

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
	size_t i, min_weights;

	assert(parent1);
	assert(parent2);

	/* Take the most fit parent as the base */
	if(parent2->fitness > parent1->fitness){
		const struct neat_genome *tmp;

		tmp = parent1;
		parent1 = parent2;
		parent2 = tmp;
	}

	child = neat_genome_copy(parent1);

	/* Iterate until the least amount of weights, if there any excess
	 * weights for the child then they are inherited automatically
	 */
	min_weights = parent1->net->nweights;
	if(parent2->net->nweights < min_weights){
		min_weights = parent2->net->nweights;
	}

	for(i = 0; i < min_weights; i++){
		int in1, in2;
		float weight1, weight2;

		in1 = parent1->innovations[i];
		in2 = parent2->innovations[i];
		if(in1 != in2){
			/* Disjoint genes will be automatically chosen from the
			 * fittest genome
			 */
			continue;
		}

		/* Matching genes */
		weight1 = parent1->net->weight[i];
		weight2 = parent2->net->weight[i];

		/* Take the average (blended crossover) */
		child->net->weight[i] = (weight1 + weight2) / 2.0f;
		/* TODO choose between average and random based
		 * on chance (uniform crossover)
		 */
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

	/* Always add a new layer if there are no hidden layers yet */
	if(genome->net->nhidden_layers == 0){
		random = 0.0f;
	}else{
		random = (float)rand() / (float)RAND_MAX;
	}
	if(random < config.genome_add_neuron_mutation_probability){
		neat_genome_add_neuron(genome,
				       innovation,
				       config.genome_default_hidden_activation,
				       config.genome_default_output_activation);
		return;
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_add_link_mutation_probability){
		neat_genome_add_link(genome, innovation);
		return;
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_change_activation_probability){
		neat_genome_mutate_activation(genome, innovation);
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

bool neat_genome_is_compatible(const struct neat_genome *genome,
			       const struct neat_genome *other,
			       float treshold,
			       size_t total_species)
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

	/* The excess is always the difference between the biggest and the
	 * smallest genome
	 */
	excess = max_weights - min_weights;
	disjoint = 0;
	/* Always add an extra one so we don't get a divide by zero */
	matching = 1;
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

	distance = 1.0f * excess / (float)max_weights;
	distance += 1.5f * disjoint / (float)max_weights;
	distance += 0.4f * weight_sum / (float)matching;

	/* Make sure there are not too many or too few species by making
	 * the treshold higher if there are already a lot of species and by 
	 * making it lower if there are already too few
	 */
	treshold *= 0.1f + (total_species / 5.0f);

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
