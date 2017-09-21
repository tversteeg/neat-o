#include "population.h"

#include <stdlib.h>

static void neat_set_best_genome(struct neat_pop *p)
{
	if(!p->genomes){
		return;
	}

	struct neat_genome *cur, **genomes = p->genomes;
	while((cur = *genomes)){
		if(p->best_genome == NULL ||
		   p->best_genome->fitness < cur->fitness){
			p->best_genome = cur;
		}
		genomes++;
	}
}

static double neat_get_fitness_criterion_max(struct neat_pop *p)
{
	if(!p->genomes){
		return -1.0;
	}

	double f = 0.0;
	struct neat_genome *cur, **genomes = p->genomes;
	while((cur = *genomes)){
		if(f < cur->fitness){
			f = cur->fitness;
		}
		genomes++;
	}

	return f;
}

static double neat_get_fitness_criterion_min(struct neat_pop *p)
{
	if(!p->genomes){
		return -1.0;
	}

	double f = 1.0;
	struct neat_genome *cur, **genomes = p->genomes;
	while((cur = *genomes)){
		if(f > cur->fitness){
			f = cur->fitness;
		}
		genomes++;
	}

	return f;
}

static double neat_get_fitness_criterion_mean(struct neat_pop *p)
{
	if(!p->genomes){
		return -1.0;
	}

	double total = 0.0;
	int count = 0;
	struct neat_genome *cur, **genomes = p->genomes;
	while((cur = *genomes)){
		total += cur->fitness;
		count++;
		genomes++;
	}

	return total / (double)count;
}

neat_pop_t neat_population_create(struct neat_config config)
{
	struct neat_pop *p = malloc(sizeof(struct neat_pop));

	p->conf = config;

	return p;
}

neat_genome_t neat_run(neat_pop_t population,
		       void(*fitness_func)(neat_genome_t *genomes),
		       int generations)
{
	struct neat_pop *p = population;

	if(!p->genomes){
		return NULL;
	}

	int i;
	for(i = 0; i < generations; i++){
		fitness_func((neat_genome_t*)p->genomes);
	}

	neat_set_best_genome(p);

	double f;
	switch(p->conf.fitness_criterion){
		case NEAT_FITNESS_CRITERION_MAX:
			f = neat_get_fitness_criterion_max(p);
			break;
		case NEAT_FITNESS_CRITERION_MIN:
			f = neat_get_fitness_criterion_min(p);
			break;
		case NEAT_FITNESS_CRITERION_MEAN:
			f = neat_get_fitness_criterion_mean(p);
			break;
		default:
			return NULL;
	}

	p->solved = (f >= p->conf.fitness_treshold);

	return NULL;
}

bool neat_is_solved(neat_pop_t population)
{
	return ((struct neat_pop*)population)->solved;
}
