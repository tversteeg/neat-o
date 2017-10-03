#include <nn.h>

#include "greatest.h"

TEST nn_create_and_destroy()
{
	struct nn_ffnet *net = nn_ffnet_create(2, 1, 2, 1);
	ASSERT(net);

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_randomize()
{
	struct nn_ffnet *net = nn_ffnet_create(2, 1, 2, 1);
	ASSERT(net);

	nn_ffnet_randomize(net);

	ASSERT(net->weight[0] != 0.0);

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_copy()
{
	struct nn_ffnet *net = nn_ffnet_create(1, 0, 1, 0);
	ASSERT(net);

	nn_ffnet_randomize(net);

	struct nn_ffnet *copy = nn_ffnet_copy(net);
	ASSERT(copy);

	for(int i = 0; i < net->nweights; i++){
		ASSERT_EQ_FMT(net->weight[i], copy->weight[i], "%g");
	}

	nn_ffnet_destroy(copy);
	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_run()
{
	double input[] = {1};

	struct nn_ffnet *net = nn_ffnet_create(1, 0, 1, 0);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_SIGMOID,
				 NN_ACTIVATION_SIGMOID);

	double *results = nn_ffnet_run(net, input);
	ASSERT(results);

	ASSERT_EQ_FMT(0.5, results[0], "%g");

	nn_ffnet_destroy(net);

	net = nn_ffnet_create(1, 2, 1, 2);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_SIGMOID);

	results = nn_ffnet_run(net, input);
	ASSERT(results);

	ASSERT_EQ_FMT(0.5, results[0], "%g");

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_run_relu()
{
	struct nn_ffnet *net = nn_ffnet_create(3, 0, 3, 0);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	double input[] = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
	double *results = nn_ffnet_run(net, input);
	ASSERT(results);

	for(int i = 0; i < net->nweights; i++){
		net->weight[i] = 1.0;
	}

	ASSERT_EQ_FMT(0.0, results[0], "%g");
	for(int i = 1; i < sizeof(input) / sizeof(double); i++){
		ASSERT_EQ_FMT(input[i], results[i], "%g");
		
	}

	nn_ffnet_destroy(net);

	PASS();
}

SUITE(nn_general)
{
	RUN_TEST(nn_create_and_destroy);
	RUN_TEST(nn_randomize);
	RUN_TEST(nn_copy);
	RUN_TEST(nn_run);
	RUN_TEST(nn_run_relu);
}

TEST nn_xor()
{
	struct nn_ffnet *net = nn_ffnet_create(2, 2, 1, 1);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_SIGMOID,
				 NN_ACTIVATION_SIGMOID);

	nn_ffnet_randomize(net);

	const double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	const double output[4] = {0, 1, 1, 0};

	for(int i = 0; i < 300; i++){
		for(int j = 0; j < 4; j++){
			nn_ffnet_train(net, input[j], output + j, 3);
		}
	}

	for(int i = 0; i < 4; i++){
		double *results = nn_ffnet_run(net, input[i]);
		ASSERT(results);

		ASSERT_IN_RANGE(output[i], results[0], 0.1);
	}

	PASS();
}

SUITE(nn_backpropagation_xor)
{
	RUN_TEST(nn_xor);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv)
{
	GREATEST_MAIN_BEGIN();

	RUN_SUITE(nn_general);
	RUN_SUITE(nn_backpropagation_xor);

	GREATEST_MAIN_END();

	return 0;
}

#if 0

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

		.population_size = 100
	};

	neat_pop_t pop = neat_population_create(conf);

	neat_run(pop, calculate_fitness, 100);

	neat_population_destroy(pop);
}
#endif
