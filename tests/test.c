#include <stdio.h>

#include <neat.h>

static double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
static double outputs[4] = {0.0, 1.0, 1.0, 0.0};

static void eval_genomes(neat_genome_t *genomes)
{
	neat_genome_t genome;
	while((genome = *genomes)){
		neat_ffnet_t net = neat_ffnet_create(genome);

		int i;
		for(i = 0; i < 4; i++){
			neat_ffnet_t output = neat_ffnet_activate(net,
								  inputs[i],
								  2);

			double out = neat_ffnet_get_output_at_index(output, 0);
			double fitness = out - outputs[i];
			neat_genome_decrease_fitness(genome, fitness * fitness);
		}

		genomes++;
	}
}

int main(int argc, char *argv[])
{
	struct neat_config conf = {
		.fitness_criterion = NEAT_FITNESS_CRITERION_MEAN,
		.input_genome_topo = 2,
		.output_genome_topo = 1
	};

	neat_pop_t pop = neat_population_create(conf);

	neat_genome_t winner = neat_run(pop, eval_genomes, 300);

	neat_ffnet_t winner_net = neat_ffnet_create(pop);
	int i;
	for(i = 0; i < 4; i++){
		neat_ffnet_t output = neat_ffnet_activate(winner_net,
							  inputs[i],
							  2);

		double output_val = neat_ffnet_get_output_at_index(output, 0);
		printf("Output %d: expected %f, got %f.\n",
		       i, outputs[i], output_val);
	}

	return 0;
}
