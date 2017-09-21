#pragma once

#include <neat.h>

#include "genome.h"

struct neat_pop{
	struct neat_config conf;

	bool solved;

	int input_genome_topo, output_genome_topo;

	struct neat_genome **genomes, *best_genome;
};
