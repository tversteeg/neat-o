#include <nn.h>
#include <neat.h>

#include "greatest.h"

const float xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
const float xor_outputs[4] = {0, 1, 1, 0};

TEST neat_create_and_destroy()
{
	struct neat_config config = {
		.network_inputs = 1,
		.network_outputs = 1,
		.population_size = 1
	};
	neat_t neat = neat_create(config);
	ASSERT(neat);

	neat_destroy(neat);
	PASS();
}

TEST neat_xor()
{
	struct neat_config config = {
		.network_inputs = 2,
		.network_outputs = 1,
		.network_hidden_nodes = 10,
		.network_hidden_layers = 4,
		.population_size = 20
	};
	neat_t neat = neat_create(config);
	ASSERT(neat);

	/* Epochs */
	for(int i = 0; i < 10000; i++){
		/* Organisms */
		for(int j = 0; j < config.population_size; j++){
			/* XOR sets */
			for(int k = 0; k < 4; k++){
				const float *results = neat_run(neat,
								j,
								xor_inputs[k]);
				ASSERT(results);
			}
		}
	}

	neat_destroy(neat);
	PASS();
}

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
	float input = 1;

	struct nn_ffnet *net = nn_ffnet_create(1, 0, 1, 0);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_SIGMOID,
				 NN_ACTIVATION_SIGMOID);

	/* Set the bias to zero and the weight to 1.0 to 
	 * easily calculate the result */
	nn_ffnet_set_bias(net, 0.0);
	for(int i = 0; i < net->nweights; i++){
		net->weight[i] = 1.0;
	}

	float *results = nn_ffnet_run(net, &input);
	ASSERT(results);

	/* The sigmoid of 1.0 should be ~0.73 */
	ASSERT_IN_RANGE(0.73, results[0], 0.1);

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_run_relu()
{
	struct nn_ffnet *net = nn_ffnet_create(1, 0, 1, 0);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	/* Set the bias to zero and the weight to 1.0 to 
	 * easily calculate the result */
	nn_ffnet_set_bias(net, 0.0);
	for(int i = 0; i < net->nweights; i++){
		net->weight[i] = 1.0;
	}

	float input[] = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
	float expected_output[] = {0.0, 0.0, 1.0, 2.0, 3.0, 4.0};

	for(int i = 0; i < sizeof(input) / sizeof(float); i++){
		float *results = nn_ffnet_run(net, input + i);
		ASSERT(results);

		ASSERT_EQ_FMT(expected_output[i], results[0], "%g");
	}

	nn_ffnet_destroy(net);

	PASS();
}

TEST nn_run_xor()
{
	struct nn_ffnet *net = nn_ffnet_create(2, 2, 1, 1);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	/* From left to right: bias, left, right
	 * From top to bottom: hidden node 1, hidden node 2 and output */
	const float weights[] = { 0.0, -1.0, 1.0,
		0.0, 1.0, -1.0,
		0.0, 1.0, 1.0 };
	memcpy(net->weight, weights, sizeof(weights));

	for(int i = 0; i < 4; i++){
		float *results = nn_ffnet_run(net, xor_inputs[i]);
		ASSERT(results);

		ASSERT_EQ_FMT(xor_outputs[i], results[0], "%g");
	}

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_time_big()
{
	struct nn_ffnet *net = nn_ffnet_create(256, 128, 20, 10);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	const float inputs[1024] = { 1.0 };

	for(int i = 0; i < 100; i++){
		nn_ffnet_run(net, inputs);
	}

	nn_ffnet_destroy(net);
	PASS();
}

SUITE(nn)
{
	RUN_TEST(nn_create_and_destroy);
	RUN_TEST(nn_randomize);
	RUN_TEST(nn_copy);
	RUN_TEST(nn_run);
	RUN_TEST(nn_run_relu);
	RUN_TEST(nn_run_xor);
}

SUITE(nn_time)
{
	RUN_TEST(nn_time_big);
}

SUITE(neat)
{
	RUN_TEST(neat_create_and_destroy);
	RUN_TEST(neat_xor);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv)
{
	srand(time(NULL));

	GREATEST_MAIN_BEGIN();

	RUN_SUITE(nn);
	RUN_SUITE(nn_time);
	RUN_SUITE(neat);

	GREATEST_MAIN_END();

	return 0;
}

#if 0

#include <neat.h>

static float xor_inputs[4][2] = {{0.0, 0.0},
	{0.0, 1.0},
	{1.0, 0.0},
	{1.0, 1.0}};
static float xor_outputs[4] = {0.0, 1.0, 1.0, 0.0};

static float calculate_fitness(neat_ffnet_t net)
{
	float fitness = 4.0;
	for(int i = 0; i < 4; i++){
		neat_ffnet_predict(net, xor_inputs[i]);
		float *outputs = neat_ffnet_get_outputs(net);
		neat_ffnet_reset(net);

		float diff = outputs[0] - xor_outputs[i];
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
