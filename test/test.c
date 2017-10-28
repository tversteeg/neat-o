#include <nn.h>
#include <neat.h>

#include <float.h>
#include <math.h>

#include "greatest.h"

const float xor_inputs[4][2] = {
	{0.0f, 0.0f},
	{0.0f, 1.0f},
	{1.0f, 0.0f},
	{1.0f, 1.0f}
};
const float xor_outputs[4] = {0.0f, 1.0f, 1.0f, 0.0f};

TEST neat_create_and_destroy()
{
	struct neat_config config;
	neat_t neat;

	config.network_inputs = 1;
	config.network_outputs = 1;
	config.network_hidden_nodes = 1;
	config.population_size = 1;
	config.minimum_time_before_replacement = 1;

	neat = neat_create(config);
	ASSERT(neat);

	neat_destroy(neat);
	PASS();
}

TEST neat_xor()
{
	neat_t neat;
	struct neat_config config;
	size_t i;
	size_t mutations;

	config.network_inputs = 2;
	config.network_outputs = 1;
	config.network_hidden_nodes = 16;

	config.population_size = 20;

	config.minimum_time_before_replacement = 1;

	config.species_crossover_probability = 0.2;
	config.interspecies_crossover_probability = 0.05;
	config.mutate_species_crossover_probability = 0.25;

	config.genome_add_neuron_mutation_probability = 0.5;
	config.genome_add_link_mutation_probability = 0.1;

	config.genome_minimum_ticks_alive = 20;
	config.genome_compatibility_treshold = 0.2;

	neat = neat_create(config);
	ASSERT(neat);

	mutations = 0;

	/* Epochs */
	for(i = 0; i < 10000; i++){
		size_t j;

		/* Organisms */
		for(j = 0; j < config.population_size; j++){
			float fitness, error;
			int k;

			/* XOR sets */
			error = 0.0f;
			for(k = 0; k < 4; k++){
				const float *results;

				results = neat_run(neat, j, xor_inputs[k]);
				ASSERT(results);

				error += fabs(results[0] - xor_outputs[k]);
			}

			if(error < 0.1){
				char message[512];

				neat_destroy(neat);

				sprintf(message,
					"Found solution after %d iterations",
					(int)i);
				PASSm(message);
			}

			fitness = 4.0 - error;
			neat_set_fitness(neat, j, fitness * fitness);

			neat_increase_time_alive(neat, j);
		}

		if(neat_epoch(neat, NULL)){
			mutations++;
		}
	}

	neat_destroy(neat);
	printf("Mutations: %d\n", (int)mutations);
	FAILm("A mutation that solved the xor problem did not occur");
}

TEST nn_create_and_destroy()
{
	struct nn_ffnet *net;

	net = nn_ffnet_create(2, 1, 2, 1);
	ASSERT(net);

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_randomize()
{
	struct nn_ffnet *net;

	net = nn_ffnet_create(2, 1, 2, 1);
	ASSERT(net);

	nn_ffnet_randomize(net);

	ASSERT(net->weight[0] != 0.0f);

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_copy_weights()
{
	struct nn_ffnet *net, *copy;
	size_t i;

	net = nn_ffnet_create(10, 10, 10, 10);
	ASSERT(net);

	nn_ffnet_set_weights(net, 1.0f);

	copy = nn_ffnet_copy(net);
	ASSERT(copy);

	/* Make sure the copies without changes are the same */
	for(i = 0; i < net->nweights; i++){
		ASSERT_EQ_FMT(net->weight[i], copy->weight[i], "%g");
	}

	/* Make sure the copies with changes are not the same */
	nn_ffnet_set_weights(net, 0.0f);
	for(i = 0; i < net->nweights; i++){
		ASSERT_FALSE(net->weight[i] == copy->weight[i]);
	}

	nn_ffnet_destroy(copy);
	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_copy_neurons()
{
	const float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	struct nn_ffnet *net, *copy;
	float *results, *results_copy;
	size_t i;

	net = nn_ffnet_create(10, 3, 10, 2);
	ASSERT(net);

	nn_ffnet_randomize(net);

	results = nn_ffnet_run(net, input);

	copy = nn_ffnet_copy(net);
	ASSERT(copy);

	/* Make sure the copies without changes are the same */
	results_copy = nn_ffnet_run(copy, input);
	for(i = 0; i < 10; i++){
		ASSERT_EQ_FMT(results[i], results_copy[i], "%g");
	}

	nn_ffnet_destroy(copy);
	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_neuron_is_connected()
{
	struct nn_ffnet *net;
	size_t i;

	net = nn_ffnet_create(4, 2, 1, 2);
	ASSERT(net);

	/* Inputs must always be true */
	for(i = 0; i < 4; i++){
		ASSERT(nn_ffnet_neuron_is_connected(net, i));
	}

	/* The rest must be false because we didn't set any weights */
	for(i = 4; i < net->nneurons; i++){
		ASSERT_FALSE(nn_ffnet_neuron_is_connected(net, i));
	}

	/* We set one weight in the first layer so the first hidden layer
	 * must be true
	 */
	for(i = 0; i < 4; i++){
		net->weight[5 * 0 + i + 1] = 1.0f;
		ASSERT(nn_ffnet_neuron_is_connected(net, 4));

		/* Reset */
		nn_ffnet_set_weights(net, 0.0f);

		net->weight[5 * 1 + i + 1] = 1.0f;
		ASSERT(nn_ffnet_neuron_is_connected(net, 5));

		/* Reset */
		nn_ffnet_set_weights(net, 0.0f);
	}

	/* Do the same for the next hidden layer */
	for(i = 0; i < 2; i++){
		net->weight[5 * 2 + 3 * 0 + i + 1] = 1.0f;
		ASSERT(nn_ffnet_neuron_is_connected(net, 6));

		/* Reset */
		nn_ffnet_set_weights(net, 0.0f);

		net->weight[5 * 2 + 3 * 1 + i + 1] = 1.0f;
		ASSERT(nn_ffnet_neuron_is_connected(net, 7));

		/* Reset */
		nn_ffnet_set_weights(net, 0.0f);
	}

	/* Do the same for the output node */
	for(i = 0; i < 2; i++){
		net->weight[5 * 2 + 3 * 2 + i + 1] = 1.0f;
		ASSERT(nn_ffnet_neuron_is_connected(net, 8));

		/* Reset */
		nn_ffnet_set_weights(net, 0.0f);
	}

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_add_layer_zero()
{
	const float inputs[] = {1.0f, 10.25f, 0.01f};

	struct nn_ffnet *net;
	float *results;
	size_t i;

	net = nn_ffnet_create(1, 1, 1, 0);
	ASSERT(net);

	/* Biases will be 0.0 */
	net->weight[1] = 1.0f;
	net = nn_ffnet_add_hidden_layer(net, 2.0f);
	ASSERT_EQ_FMT(0.0f, net->weight[0], "%g");
	ASSERT_EQ_FMT(2.0f, net->weight[1], "%g");
	ASSERT_EQ_FMT(0.0f, net->weight[2], "%g");
	ASSERT_EQ_FMT(1.0f, net->weight[3], "%g");

	nn_ffnet_destroy(net);

	net = nn_ffnet_create(3, 3, 3, 0);
	ASSERT(net);

	net->weight[1] = 1.0f;
	net->weight[6] = 1.0f;
	net->weight[11] = 1.0f;

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	nn_ffnet_set_bias(net, 0.0f);

	net = nn_ffnet_add_hidden_layer(net, 1.0f);

	results = nn_ffnet_run(net, inputs);
	ASSERT(results);

	for(i = 0; i < 3; i++){
		ASSERT_IN_RANGE(inputs[i], results[i], 0.01f);
	}

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_add_layer_single()
{
	const float input = 1;

	struct nn_ffnet *net;
	float *results;

	net = nn_ffnet_create(1, 1, 1, 1);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	/* Set the input -> hidden & hidden -> output layers to 1.0 */
	nn_ffnet_set_bias(net, 0.0);
	net->weight[1] = 1.0f;
	net->weight[3] = 2.0f;

	net = nn_ffnet_add_hidden_layer(net, 3.0f);

	results = nn_ffnet_run(net, &input);
	ASSERT(results);

	ASSERT_EQ_FMT(6.0, results[0], "%g");

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_add_layer_multi()
{
	struct nn_ffnet *net, *copy;
	size_t i;
	int j;

	net = nn_ffnet_create(2, 2, 2, 2);
	ASSERT(net);

	nn_ffnet_randomize(net);

	copy = nn_ffnet_copy(net);

	net = nn_ffnet_add_hidden_layer(net, 2.0f);

	/* Compare the inputs and the first hidden layers */
	for(i = 0; i < 6; i++){
		ASSERT_EQ_FMT(copy->weight[i], net->weight[i], "%g");
	}

	/* Compare the outputs */
	for(j = -1; j >= -2; j--){
		ASSERT_EQ_FMT(copy->output[j], net->output[j], "%g");
	}

	nn_ffnet_destroy(net);
	nn_ffnet_destroy(copy);
	PASS();
}

TEST nn_run()
{
	const float input = 1;

	struct nn_ffnet *net;
	float *results;
	size_t i;

	net = nn_ffnet_create(1, 1, 1, 0);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_SIGMOID,
				 NN_ACTIVATION_SIGMOID);

	/* Set the bias to zero and the weight to 1.0 to 
	 * easily calculate the result */
	nn_ffnet_set_bias(net, 0.0);
	for(i = 0; i < net->nweights; i++){
		net->weight[i] = 1.0;
	}

	results = nn_ffnet_run(net, &input);
	ASSERT(results);

	/* The sigmoid of 1.0 should be ~0.73 */
	ASSERT_IN_RANGE(0.73, results[0], 0.1);

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_run_relu()
{
	const float input[] = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
	const float expected_output[] = {0.0, 0.0, 1.0, 2.0, 3.0, 4.0};

	struct nn_ffnet *net;
	size_t i;

	net = nn_ffnet_create(1, 1, 1, 0);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	/* Set the bias to zero and the weight to 1.0 to 
	 * easily calculate the result */
	nn_ffnet_set_bias(net, 0.0);
	for(i = 0; i < net->nweights; i++){
		net->weight[i] = 1.0;
	}

	for(i = 0; i < sizeof(input) / sizeof(float); i++){
		float *results;

		results = nn_ffnet_run(net, input + i);
		ASSERT(results);

		ASSERT_EQ_FMT(expected_output[i], results[0], "%g");
	}

	nn_ffnet_destroy(net);

	PASS();
}

TEST nn_run_xor()
{
	/* From left to right: bias, left, right
	 * From top to bottom: hidden node 1, hidden node 2 and output */
	const float weights[] = { 0.0, -1.0, 1.0,
		0.0, 1.0, -1.0,
		0.0, 1.0, 1.0 };

	struct nn_ffnet *net;
	size_t i;

	net = nn_ffnet_create(2, 2, 1, 1);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);
	memcpy(net->weight, weights, sizeof(weights));

	for(i = 0; i < 4; i++){
		float *results;

		results = nn_ffnet_run(net, xor_inputs[i]);
		ASSERT(results);

		ASSERT_EQ_FMT(xor_outputs[i], results[0], "%g");
	}

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_time_big()
{
	const float inputs[1024] = { 1.0 };

	struct nn_ffnet *net;
	size_t i;

	net = nn_ffnet_create(1024, 256, 64, 4);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	for(i = 0; i < 100; i++){
		nn_ffnet_run(net, inputs);
	}

	nn_ffnet_destroy(net);
	PASS();
}

SUITE(nn)
{
	size_t i;

	for(i = 0; i < 10; i++){
		RUN_TEST(nn_create_and_destroy);
		RUN_TEST(nn_randomize);
		RUN_TEST(nn_copy_weights);
		RUN_TEST(nn_copy_neurons);
		RUN_TEST(nn_neuron_is_connected);
		RUN_TEST(nn_add_layer_zero);
		RUN_TEST(nn_add_layer_single);
		RUN_TEST(nn_add_layer_multi);
		RUN_TEST(nn_run);
		RUN_TEST(nn_run_relu);
		RUN_TEST(nn_run_xor);
	}
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
