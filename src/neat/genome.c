#include "genome.h"

#include <string.h>
#include <assert.h>

struct neat_genome *neat_genome_create(struct neat_config config,
				       int innovation)
{
	assert(innovation > 0);

	struct neat_genome *genome = calloc(1, sizeof(struct neat_genome));
	assert(genome);

	genome->net = nn_ffnet_create(config.network_inputs,
				      0,
				      config.network_outputs,
				      0);

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

	size_t bytes = sizeof(int) * genome->net->nweights;
	new->innovations = malloc(bytes);
	memcpy(new->innovations, genome->innovations, bytes);

	return new;
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

}

bool neat_genome_is_compatible(const struct neat_genome *genome,
			       const struct neat_genome *other,
			       float treshold)
{
	//TODO implement by checking the distance between the genomes

	return true;
}
