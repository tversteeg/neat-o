#include "genome.h"

#include <string.h>
#include <assert.h>

static size_t neat_genome_allocate_innovations(struct neat_genome *genome)
{
	assert(genome);
	assert(genome->net);

	//TODO replace this with a copying of the values function
	if(genome->innovations){
		free(genome->innovations);
	}

	size_t bytes = sizeof(int) * genome->net->nweights;
	genome->innovations = malloc(bytes);
	assert(genome->innovations);

	return bytes;
}

static struct nn_ffnet *neat_nn_add_layer(const struct nn_ffnet *net)
{
	struct nn_ffnet *new = nn_ffnet_create(net->ninputs,
					       net->nhiddens,
					       net->noutputs,
					       net->nhidden_layers + 1);
	assert(new);
	
	//TODO copy internal data

	return new;
}

static void neat_genome_add_neuron(struct neat_genome *genome)
{
	assert(genome);
	assert(genome->net);

	/* Add + 1 to the selection of the layer so a new one can be created */
	size_t layer = rand() % (genome->net->nhidden_layers + 1);
	if(layer >= genome->net->nhidden_layers){
		struct nn_ffnet *new = neat_nn_add_layer(genome->net);
		nn_ffnet_destroy(genome->net);
		genome->net = new;

		neat_genome_allocate_innovations(genome);
	}
}

static void neat_genome_add_link(struct neat_genome *genome)
{
	assert(genome);

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

	genome->innovations = malloc(sizeof(int) * genome->net->nweights);
	assert(genome->innovations);
	for(size_t i = 0; i < genome->net->nweights; i++){
		genome->innovations[i] = innovation;
	}

	return genome;
}

struct neat_genome *neat_genome_copy(const struct neat_genome *genome)
{
	struct neat_genome *new = malloc(sizeof(struct neat_genome));
	assert(new);

	memcpy(new, genome, sizeof(struct neat_genome));

	new->net = nn_ffnet_copy(genome->net);
	assert(new->net);

	new->innovations = NULL;
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

void neat_genome_mutate(struct neat_genome *genome, struct neat_config config)
{
	assert(genome);

	float random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_add_neuron_mutation_probability){
		neat_genome_add_neuron(genome);
		return;
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < config.genome_add_link_mutation_probability){
		neat_genome_add_link(genome);
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
	assert(innovation >= 0);

	//TODO implement
}

bool neat_genome_is_compatible(const struct neat_genome *genome,
			       const struct neat_genome *other,
			       float treshold)
{
	//TODO implement by checking the distance between the genomes

	return true;
}

