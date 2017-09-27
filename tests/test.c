#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <neat.h>

static double xor_inputs[4][2] = {{0.0, 0.0},
				  {0.0, 1.0},
				  {1.0, 0.0},
				  {1.0, 1.0}};
static double xor_outputs[4] = {0.0, 1.0, 1.0, 0.0};

static double calculate_fitness(neat_ffnet_t net)
{
	double fitness = 4.0;
	for(int i = 0; i < 4; i++){
		neat_ffnet_predict(net, xor_inputs[i]);
		double *outputs = neat_ffnet_get_outputs(net);
		neat_ffnet_reset(net);

		double diff = outputs[0] - xor_outputs[i];
		fitness -= diff * diff;
		free(outputs);
	}

	return fitness;
}

static void run()
{
	/* Initialize random numbers */
	srand(time(NULL));

	struct neat_config conf = {
		.fitness_criterion = NEAT_FITNESS_CRITERION_MEAN,
		.input_genome_topo = 2,
		.output_genome_topo = 1,

		.population_size = 15
	};

	neat_pop_t pop = neat_population_create(conf);

	neat_run(pop, calculate_fitness, 100);

	neat_population_destroy(pop);
}

int main(int argc, char *argv[])
{
	run();
	return 0;
}
