#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <neat.h>

static double xor_inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
static double xor_outputs[4] = {0.0, 1.0, 1.0, 0.0};

static int gen = 0;

static double calculate_fitness(double *outputs)
{
	return outputs[0] - xor_outputs[gen];
}

int main(int argc, char *argv[])
{
	/* Initialize random numbers */
	srand(time(NULL));

	struct neat_config conf = {
		.fitness_criterion = NEAT_FITNESS_CRITERION_MEAN,
		.input_genome_topo = 2,
		.output_genome_topo = 1,

		.population_size = 100
	};

	neat_pop_t pop = neat_population_create(conf);

	for(gen = 0; gen < 4; gen++){
		printf("Inputs %d: %f, %f\n", xor_inputs[gen][0], xor_inputs[gen][1]);
		neat_run(pop, xor_inputs[gen], calculate_fitness, 100);
	}

	return 0;
}
