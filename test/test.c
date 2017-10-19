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
	struct neat_config config = {
		.network_inputs = 1,
		.network_outputs = 1,
		.network_hidden_nodes = 1,
		.network_hidden_layers = 1,
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
		.network_hidden_nodes = 16,
		.network_hidden_layers = 1,

		.population_size = 20,

		.species_crossover_probability = 0.2,
		.interspecies_crossover_probability = 0.05,
		.mutate_species_crossover_probability = 0.25,

		.genome_add_neuron_mutation_probability = 0.5,
		.genome_add_link_mutation_probability = 0.1,

		.genome_minimum_ticks_alive = 100,
		.genome_compatibility_treshold = 0.2
	};
	neat_t neat = neat_create(config);
	ASSERT(neat);

	/* Epochs */
	for(int i = 0; i < 10000; i++){
		/* Organisms */
		for(int j = 0; j < config.population_size; j++){
			/* XOR sets */
			float error = 0.0f;
			for(int k = 0; k < 4; k++){
				const float *results = neat_run(neat,
								j,
								xor_inputs[k]);
				ASSERT(results);

				error += fabs(results[0] - xor_outputs[k]);
			}

			if(error < 0.1){
				neat_destroy(neat);

				char message[512];
				snprintf(message,
					 512,
					 "Found solution after %d iterations",
					 i);
				PASSm(message);
			}

			float fitness = 4.0 - error;
			neat_set_fitness(neat, j, fitness * fitness);

			neat_increase_time_alive(neat, j);
		}

		neat_epoch(neat);
	}

	/* Organisms */
	for(int j = 0; j < config.population_size; j++){
		//neat_print_net(neat, j);
	}

	neat_destroy(neat);
	FAILm("A mutation that solved the xor problem did not occur");
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

TEST nn_copy_weights()
{
	struct nn_ffnet *net = nn_ffnet_create(10, 10, 10, 10);
	ASSERT(net);

	nn_ffnet_set_weights(net, 1.0f);

	struct nn_ffnet *copy = nn_ffnet_copy(net);
	ASSERT(copy);

	/* Make sure the copies without changes are the same */
	for(int i = 0; i < net->nweights; i++){
		ASSERT_EQ_FMT(net->weight[i], copy->weight[i], "%g");
	}

	/* Make sure the copies with changes are not the same */
	nn_ffnet_set_weights(net, 0.0f);
	for(int i = 0; i < net->nweights; i++){
		ASSERT_FALSE(net->weight[i] == copy->weight[i]);
	}

	nn_ffnet_destroy(copy);
	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_copy_neurons()
{
	float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	struct nn_ffnet *net = nn_ffnet_create(10, 3, 10, 2);
	ASSERT(net);

	nn_ffnet_randomize(net);

	float *results = nn_ffnet_run(net, input);

	struct nn_ffnet *copy = nn_ffnet_copy(net);
	ASSERT(copy);

	/* Make sure the copies without changes are the same */
	float *results_copy = nn_ffnet_run(copy, input);
	for(int i = 0; i < 10; i++){
		ASSERT_EQ_FMT(results[i], results_copy[i], "%g");
	}

	nn_ffnet_destroy(copy);
	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_neuron_is_connected()
{
	struct nn_ffnet *net = nn_ffnet_create(4, 2, 1, 2);
	ASSERT(net);

	/* Inputs must always be true */
	for(size_t i = 0; i < 4; i++){
		ASSERT(nn_ffnet_neuron_is_connected(net, i));
	}

	/* The rest must be false because we didn't set any weights */
	for(size_t i = 4; i < net->nneurons; i++){
		ASSERT_FALSE(nn_ffnet_neuron_is_connected(net, i));
	}

	/* We set one weight in the first layer so the first hidden layer
	 * must be true
	 */
	for(size_t i = 0; i < 4; i++){
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
	for(size_t i = 0; i < 2; i++){
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
	for(size_t i = 0; i < 2; i++){
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
	float inputs[] = {1.0f, 10.25f, 0.01f};

	struct nn_ffnet *net = nn_ffnet_create(3, 3, 3, 0);
	ASSERT(net);

	net->weight[1] = 1.0f;
	net->weight[6] = 1.0f;
	net->weight[11] = 1.0f;

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	nn_ffnet_set_bias(net, 0.0f);

	net = nn_ffnet_add_hidden_layer(net, 1.0f);

#if 0
	puts("\n");
	for(size_t i = 0; i < net->nweights; i++){
		if(i % 4 == 0){
			puts("\n");
		}
		printf("%X:%f ",
		       (unsigned)(unsigned long long)(net->weight + i),
		       net->weight[i]);
	}
	puts("\n");
#endif

	float *results = nn_ffnet_run(net, inputs);
	ASSERT(results);

	for(size_t i = 0; i < 3; i++){
		ASSERT_IN_RANGE(inputs[i], results[i], 0.01f);
	}

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_add_layer_single()
{
	float input = 1;

	struct nn_ffnet *net = nn_ffnet_create(1, 1, 1, 1);
	ASSERT(net);

	nn_ffnet_set_activations(net,
				 NN_ACTIVATION_RELU,
				 NN_ACTIVATION_RELU);

	/* Set the input -> hidden & hidden -> output layers to 1.0 */
	nn_ffnet_set_bias(net, 0.0);
	net->weight[1] = 1.0f;
	net->weight[3] = 2.0f;

	net = nn_ffnet_add_hidden_layer(net, 1.0f);

	float *results = nn_ffnet_run(net, &input);
	ASSERT(results);

	ASSERT_EQ_FMT(2.0, results[0], "%g");

	nn_ffnet_destroy(net);
	PASS();
}

TEST nn_add_layer_multi()
{
	struct nn_ffnet *net = nn_ffnet_create(2, 2, 2, 2);
	ASSERT(net);

	nn_ffnet_randomize(net);

	struct nn_ffnet *copy = nn_ffnet_copy(net);

	net = nn_ffnet_add_hidden_layer(net, 2.0f);

	/* Compare the inputs and the first hidden layers */
	for(size_t i = 0; i < 6; i++){
		ASSERT_EQ_FMT(copy->weight[i], net->weight[i], "%g");
	}

	/* Compare the outputs */
	for(size_t i = -1; i >= -2; i--){
		ASSERT_EQ_FMT(copy->output[i], net->output[i], "%g");
	}

	nn_ffnet_destroy(net);
	nn_ffnet_destroy(copy);
	PASS();
}

TEST nn_run()
{
	float input = 1;

	struct nn_ffnet *net = nn_ffnet_create(1, 1, 1, 0);
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
	struct nn_ffnet *net = nn_ffnet_create(1, 1, 1, 0);
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
	struct nn_ffnet *net = nn_ffnet_create(1024, 256, 64, 4);
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
	for(size_t i = 0; i < 10; i++){
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
