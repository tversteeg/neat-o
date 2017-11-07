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

	/* Set the weight innovations to 0 if the value of the weight is 0.0 */
	genome->used_weights = 0;
	for(i = 0; i < genome->ninnov_weights; i++){
		int weight_is_set;

		/* Set the innovation to 0 if the weight is 0 */
		weight_is_set = genome->net->weight[i] != 0;
		genome->innov_weight[i] *= weight_is_set;
		genome->used_weights += weight_is_set;
	}

	/* Set the weight innovations to 0 if the value of the weight is 0.0 */
	genome->used_activs = 0;
	for(i = 0; i < genome->ninnov_activs; i++){
		int activ_is_not_passthrough;

		/* Set the innovation to 0 if the weight is 0 */
		activ_is_not_passthrough = genome->net->activation[i] != 0;
		genome->innov_activ[i] *= activ_is_not_passthrough;
		genome->used_activs += activ_is_not_passthrough;
	}

	/*TODO make the used_activs and used_weights more sensible */
}

static size_t neat_genome_allocate_innovations(struct neat_genome *genome,
					       int innovation)
{
	int *innov;
	size_t bytes, diff, i;

	assert(genome);
	assert(genome->net);

	/* Allocate the weight innovations */
	bytes = sizeof(int) * genome->net->nweights;
	assert(bytes > 0);
	genome->innov_weight = realloc(genome->innov_weight, bytes);
	assert(genome->innov_weight);

	/* Set the newly allocated part to the current innovation */
	diff = genome->net->nweights - genome->ninnov_weights;
	innov = genome->innov_weight + genome->ninnov_weights;
	for(i = 0; i < diff; i++){
		*innov++ = innovation;
	}

	genome->ninnov_weights = genome->net->nweights;

	/* Allocate the activation innovations */
	bytes = sizeof(int) * genome->net->nactivations;
	assert(bytes > 0);
	genome->innov_activ = realloc(genome->innov_activ, bytes);
	assert(genome->innov_activ);

	/* Set the newly allocated part to the current innovation */
	diff = genome->net->nactivations - genome->ninnov_activs;
	innov = genome->innov_activ + genome->ninnov_activs;
	for(i = 0; i < diff; i++){
		*innov++ = innovation;
	}

	genome->ninnov_activs = genome->net->nactivations;

	return bytes;
}

static void neat_genome_add_layer(struct neat_genome *genome, int innovation)
{
	assert(genome);

	genome->net = nn_ffnet_add_hidden_layer(genome->net, 1.0);

	neat_genome_allocate_innovations(genome, innovation);
	neat_genome_zeroify_innovations(genome);
}

static void neat_genome_add_link(struct neat_genome *genome, int innovation)
{
	size_t available, select_weight_offset, i;

	assert(genome);
	assert(genome->net);

	/* Select a random available weight */
	available = genome->net->nweights - genome->used_weights;
	/* Do nothing if there are no more available weights */
	if(available == 0){
		return;
	}

	/* Get a random number between the start and the end of all available
	 * weights
	 */
	select_weight_offset = rand() % available;

	/* Loop over the available weight to find the randomly selected one */
	for(i = 0; i < genome->net->nweights; i++){
		/* Skip over unavailable weights */
		if(genome->net->weight[i] != 0.0f){
			continue;
		}

		/* Count down available weights until the randomly selected
		 * one is found
		 */
		if(!select_weight_offset--){
			genome->net->weight[i] = neat_random_two();
			genome->innov_weight[i] = innovation;
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
	size_t i, layer, start_offset;

	assert(genome);
	assert(genome->net);

	n = genome->net;

	/* Add + 1 to the selection of the layer so a new one can be created */
	if(n->nhidden_layers == 0){
		layer = 0;
		neat_genome_add_layer(genome, innovation);
	}else{
		layer = rand() % (n->nhidden_layers + 1);
		if(layer >= n->nhidden_layers){
			neat_genome_add_layer(genome, innovation);
		}
	}

	/* Start at the begin of the layer for the neurons */
	start_offset = n->ninputs + layer * n->nhiddens;

	/* Adda random offset so not the same vertical layer will be chosen
	 * every time
	 */
	start_offset += rand() % n->nhiddens;

	/* Find the first disconnected layer starting from the selected layer
	 * and set the weight value to a random previous neuron
	 */
	for(i = start_offset; i < n->nneurons; i++){
		size_t activ_offset;
		char *activ;

		activ_offset = i - n->ninputs;
		activ = n->activation + activ_offset;
		if(*activ == NN_ACTIVATION_PASSTHROUGH){
			if(layer == n->nhidden_layers){
				/* Set the output activation if it's the last
				 * layer
				 */
				*activ = default_output;
			}else{
				*activ = default_hidden;
			}
			genome->innov_activ[activ_offset] = innovation;
			genome->used_activs++;

			return;
		}
	}
}

static void neat_genome_mutate_activation(struct neat_genome *genome,
					  int innovation)
{
	size_t random_activ;
	char new_activation;

	assert(genome);
	assert(genome->net);

	random_activ = rand() % genome->net->nactivations;

	/* TODO make the new activation never be passthrough */
	/* Randomly select a new activation function */
	new_activation = rand() % _NN_ACTIVATION_COUNT;

	/* If the activation is the same as the last one just increment it */
	if(genome->net->activation[random_activ] == new_activation){
		new_activation = (new_activation + 1) % _NN_ACTIVATION_COUNT;
	}

	genome->net->activation[random_activ] = new_activation;
	genome->innov_activ[random_activ] = innovation;
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
			genome->innov_weight[i] = innovation;
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
			genome->innov_weight[i] = innovation;
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

	nn_ffnet_set_bias(genome->net, -1.0f);

	neat_genome_allocate_innovations(genome, innovation);
	for(i = 0; i < genome->ninnov_weights; i++){
		genome->innov_weight[i] = innovation;
	}
	for(i = 0; i < genome->ninnov_activs; i++){
		genome->innov_activ[i] = innovation;
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

	new->innov_weight = NULL;
	bytes = neat_genome_allocate_innovations(new, 1);
	memcpy(new->innov_weight, genome->innov_weight, bytes);
	assert(new->innov_weight);

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
	min_weights = parent1->ninnov_weights;
	if(parent2->ninnov_weights < min_weights){
		min_weights = parent2->ninnov_weights;
	}

	for(i = 0; i < min_weights; i++){
		int in1, in2;
		float weight1, weight2;

		in1 = parent1->innov_weight[i];
		in2 = parent2->innov_weight[i];
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

	/* TODO also do this for the activations */

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
	if(random <= config.genome_add_neuron_mutation_probability){
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
	free(genome->innov_weight);
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
	assert(genome->innov_weight);
	assert(other->innov_weight);

	weights1 = genome->ninnov_weights;
	weights2 = other->ninnov_weights;
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
		if(genome->innov_weight[i] == other->innov_weight[i]){
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
