#include <neat.h>

static double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
static double outputs[4] = {0.0, 1.0, 1.0, 0.0};

static void eval_gens(neat_gen_t *gens)
{
	neat_gen_t gen;
	while((gen = *gens)){
		neat_ffnet_t net = neat_ffnet_create(gen);

		int i;
		for(i = 0; i < 4; i++){
			neat_ffnet_t output = neat_ffnet_activate(net,
								  inputs[i],
								  2);

			double out = neat_ffnet_get_output_at_index(output, 0);
			double fitness = out - outputs[i];
			neat_gen_decrease_fitness(gen, fitness * fitness);
		}

		gens++;
	}
}

int main(int argc, char *argv[])
{
	neat_population_t pop = neat_population_create();

	neat_gen_t winner = neat_run(pop, eval_gens, 300);

	neat_ffnet_t winner_net = neat_ffnet_create(winner);
	int i;
	for(i = 0; i < 4; i++){
		neat_ffnet_t output = neat_ffnet_activate(winner_net,
							  inputs[i],
							  2);
	}

	return 0;
}
