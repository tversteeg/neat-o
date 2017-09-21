#include "genome.h"

void neat_genome_decrease_fitness(neat_genome_t genome, double fitness)
{
	struct neat_genome *g = genome;

	g->fitness -= fitness;
}
