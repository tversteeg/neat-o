#pragma once

typedef void* neat_population_t;
typedef void* neat_gen_t;
typedef void* neat_ffnet_t;

neat_population_t neat_population_create();

neat_gen_t neat_run(neat_population_t population,
		    void(*fitness_func)(neat_gen_t *gens),
		    int generations);

neat_ffnet_t neat_ffnet_create(neat_gen_t winner);
neat_ffnet_t neat_ffnet_activate(neat_ffnet_t net, double* inputs, int ninputs);

double neat_ffnet_get_output_at_index(neat_ffnet_t net, int index);

void neat_gen_decrease_fitness(neat_gen_t gen, double fitness);
