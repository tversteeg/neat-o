#include "species.h"

#include <stdlib.h>
#include <math.h>

static void neat_species_culling(struct neat_species *s,
				 double avg_fitness)
{
	if(avg_fitness > s->max_avg_fitness){
		s->max_avg_fitness = avg_fitness;
		s->generation_with_max_fitness = s->generation;
	}

	/* TODO replace with const.STAGNATED_SPECIES_TRESHOLD */
	if(s->generation - s->generation_with_max_fitness > 20){
		/* TODO replace with const.STAGNATED_SPECIES_ALLOWED */
		if(++s->times_stagnated > 2){
			s->active = false;
		}else{
			s->max_avg_fitness = 0;
			s->generation_with_max_fitness = s->generation;

			struct neat_ffnet *genome = s->genomes;
			for(int i = 1; i < s->population; i++){
				s->genomes[i] = neat_ffnet_copy(genome);
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

	s.genomes = malloc(sizeof(struct neat_ffnet) * s.population);
	for(int i = 0; i < s.population; i++){
		s.genomes[i] = neat_ffnet_copy(genome);
	}

	return s;
}

bool neat_species_run(struct neat_species *s,
		      double(*fitness_func)(double *outputs),
		      double *avg_fitness)
{
	if(!s->active){
		return false;
	}

	s->generation++;

	double fitness = 0.0;
	for(int i = 0; i < s->population; i++){
		struct neat_ffnet *genome = s->genomes + i;
		double *outputs = neat_ffnet_get_outputs(genome);
		
		fitness += fitness_func(outputs);
		free(outputs);
	}

	*avg_fitness = fitness / (double)(s->population + 1);

	neat_species_culling(s, *avg_fitness);

	return s->active;
}

void neat_species_evolve(struct neat_species *s)
{

}
