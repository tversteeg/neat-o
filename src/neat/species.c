#include "species.h"

#include <stdlib.h>
#include <math.h>
#include <assert.h>

static int neat_ffnet_fitness_compare(const void *network1,
				      const void *network2)
{
	const struct neat_ffnet *n1 = network1;
	assert(n1);

	const struct neat_ffnet *n2 = network2;
	assert(n2);
	
	double diff = n1->fitness - n2->fitness;
	double sign = (diff > 0) - (diff < 0);

	return sign;
}

static struct neat_ffnet *neat_ffnet_crossover(struct neat_ffnet *gen,
					       struct neat_ffnet *mate)
{
	struct neat_ffnet *fit = malloc(sizeof(struct neat_ffnet));
	struct neat_ffnet *unfit;
	if(gen->fitness > mate->fitness){
		neat_ffnet_copynew(fit, gen);
		unfit = mate;
	}else{
		neat_ffnet_copynew(fit, mate);
		unfit = gen;
	}

	for(int i = 0; i < fit->ngenes; i++){
		/* TODO implement crossover */
	}

	return fit;
}

static void neat_species_culling(struct neat_species *s,
				 double avg_fitness)
{
	assert(s);

	if(avg_fitness > s->max_avg_fitness){
		s->max_avg_fitness = avg_fitness;
		s->generation_with_max_fitness = s->generation;
	}

	/* TODO replace with const.STAGNATED_SPECIES_TRESHOLD */
	if(s->generation - s->generation_with_max_fitness > 20){
		/* TODO replace with const.STAGNATED_SPECIES_ALLOWED */
		if(++s->times_stagnated > 2){
			fprintf(stderr,
				"Species %d culled due to stagnations\n",
				s->id);

			s->active = false;
		}else{
			s->max_avg_fitness = 0;
			s->generation_with_max_fitness = s->generation;

			struct neat_ffnet *genome = s->genomes;
			for(int i = 1; i < s->population; i++){
				neat_ffnet_copy(s->genomes + i, genome);
				neat_ffnet_randomize_weights(s->genomes + i);
			}
		}
	}
	
	/* TODO replace with const.WEAK_SPECIES_TRESHOLD */
	if(s->population < 5){
		s->active = false;
	}
}

struct neat_species neat_species_create(struct neat_config config, int id,
					struct neat_ffnet *genome)
{
	assert(genome);
	assert(config.population_size > 0);

	struct neat_species s = {
		.generation = 0,
		.generation_with_max_fitness = 0,
		.max_avg_fitness = 0,
		.times_stagnated = 0,
		.active = true,

		.id = id,
		.population = config.population_size,
		.genome_representative = genome
	};

	genome->species_id = id;
	genome->generation = s.generation;

	s.genomes = calloc(s.population, sizeof(struct neat_ffnet));
	assert(s.genomes);
	for(int i = 0; i < s.population; i++){
		neat_ffnet_copynew(s.genomes + i, genome);
	}

	return s;
}

void neat_species_destroy(struct neat_species *species)
{
	assert(species);

	if(species->population > 0){
		for(int i = 0; i < species->population; i++){
			neat_ffnet_destroy(species->genomes + i);
		}
		free(species->genomes);
		species->genomes = NULL;
	}
}

bool neat_species_run(struct neat_species *s,
		      double(*fitness_func)(neat_ffnet_t net),
		      double *avg_fitness)
{
	assert(s);
	assert(fitness_func);
	assert(avg_fitness);

	if(!s->active){
		return false;
	}

	s->generation++;

	double fitness = 0.0;
	for(int i = 0; i < s->population; i++){
		assert(s->genomes + i);
		double genome_fitness = fitness_func(s->genomes + i);
		s->genomes[i].fitness = genome_fitness;
		fitness += genome_fitness;
	}

	*avg_fitness = fitness / (double)(s->population + 1);

	neat_species_culling(s, *avg_fitness);

	return s->active;
}

void neat_species_evolve(struct neat_species *s)
{
	assert(s);

	if(!s->active){
		return;
	}

	qsort(s->genomes, s->population, sizeof(struct neat_ffnet),
	      neat_ffnet_fitness_compare);

	size_t survivor_population = s->population / 2;
	assert(s->population > 0);

	size_t survivor_bytes = sizeof(struct neat_ffnet) * survivor_population;
	struct neat_ffnet *survivors = malloc(survivor_bytes);
	assert(survivors);

	for(size_t i = 0; i < survivor_population; i++){
		neat_ffnet_copynew(survivors + i, s->genomes + i);
	}

	/* s->genomes[0] is the champion, keep it */
	for(size_t i = 1; i < s->population; i++){
		/* TODO change to skewed random sampling */
		int new_sample = rand() % survivor_population;
		struct neat_ffnet *new_genome = survivors + new_sample;

		/* TODO change to config.CROSSOVER_CHANCE */
		if(rand() < RAND_MAX / 100){
			neat_ffnet_copy(s->genomes + i, new_genome);
			continue;
		}

		int mate_sample = rand() % survivor_population;
		struct neat_ffnet *mate = survivors + mate_sample;

		struct neat_ffnet *new = neat_ffnet_crossover(new_genome, mate);
		neat_ffnet_copy(s->genomes + i, new);
		neat_ffnet_destroy(new);
		free(new);

		neat_ffnet_mutate(s->genomes + i);
	}

	for(size_t i = 0; i < survivor_population; i++){
		neat_ffnet_destroy(survivors + i);
	}
	free(survivors);
}
