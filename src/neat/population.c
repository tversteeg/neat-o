#include "population.h"

#include <float.h>
#include <assert.h>

static void neat_reset_genomes(struct neat_pop *p)
{
	assert(p);

	/* Create a base genome and copy it for every other one */
	p->genomes[0] = neat_genome_create(p->conf, p->innovation++);

	for(size_t i = 1; i < p->ngenomes; i++){
		p->genomes[i] = neat_genome_copy(p->genomes[0]);
	}
}

static void neat_replace_genome(struct neat_pop *p,
				size_t dest,
				struct neat_genome *src)
{
	assert(p);
	assert(src);
	assert(p->genomes[dest] != src);

	neat_genome_destroy(p->genomes[dest]);
	p->genomes[dest] = neat_genome_copy(src);
}

static struct neat_species *neat_create_new_species(struct neat_pop *p,
						    bool fill)
{
	assert(p);

	p->species = realloc(p->species,
			     sizeof(struct neat_species*) * ++p->nspecies);
	assert(p->species);

	struct neat_species *new = neat_species_create(p->conf);

	if(fill){
		for(size_t i = 0; i < p->ngenomes; i++){
			neat_species_add_genome(new, p->genomes[i]);
		}
	}

	p->species[p->nspecies - 1] = new;

	return new;
}

static bool neat_find_worst_fitness(struct neat_pop *p, size_t *worst_genome)
{
	assert(p);
	assert(worst_genome);

	bool found_worst = false;

	float worst_fitness = FLT_MAX;
	for(size_t i = 0; i < p->ngenomes; i++){
		struct neat_genome *genome = p->genomes[i];

		float fitness = genome->fitness;
		if(fitness < worst_fitness &&
		   genome->time_alive > p->conf.genome_minimum_ticks_alive){
			*worst_genome = i;
			worst_fitness = fitness;
			found_worst = true;
		}
	}

	return found_worst;
}

static float neat_get_species_fitness_average(struct neat_pop *p)
{
	assert(p);

	float total_avg = 0.0;
	for(size_t i = 0; i < p->nspecies; i++){
		total_avg += neat_species_get_average_fitness(p->species[i]);
	}
	total_avg /= (double)p->nspecies;

	return total_avg;
}

static void neat_speciate_genome(struct neat_pop *p, size_t genome_id)
{
	assert(p);

	struct neat_genome *genome = p->genomes[genome_id];
	float compatibility_treshold = p->conf.genome_compatibility_treshold;

	/* Add genome to species if the representant matches the genome */
	for(size_t i = 0; i < p->nspecies; i++){
		if(p->species[i]->ngenomes == 0){
			continue;
		}
		struct neat_genome *species_representant =
			neat_species_get_representant(p->species[i]);
		if(neat_genome_is_compatible(genome,
					     species_representant,
					     compatibility_treshold)){
			neat_species_add_genome(p->species[i], genome);
			return;
		}
	}

	/* If no matching species could be found create a new species */
	struct neat_species *new = neat_create_new_species(p, false);
	neat_species_add_genome(new, genome);
}

static struct neat_genome *neat_crossover_get_parent2(struct neat_pop *p,
						      struct neat_species *s)
{
	assert(p);
	assert(s);

	float random = (float)rand() / (float)RAND_MAX;

	/* Check for interspecies mutation which won't work with
	 * only 1 species
	 */
	if(p->nspecies > 1 &&
	   random < p->conf.interspecies_crossover_probability){
		size_t species_index = rand() % p->nspecies;
		struct neat_species *random_species = p->species[species_index];

		/* We don't want to do interspecies crossover on the same
		 * species
		 */
		if(random_species == s){
			/* We can't move the index lower than 0 */
			if(p->species[0] == random_species){
				random_species++;
			}else{
				random_species--;
			}
		}

		if(random_species->ngenomes == 0){
			return NULL;
		}

		return neat_species_select_genitor(random_species);
	}else{
		return neat_species_select_genitor(s);
	}
}

static void neat_crossover(struct neat_pop *p,
			   struct neat_species *s,
			   size_t worst_genome,
			   struct neat_genome *parent)
{
	assert(p);
	assert(s);
	assert(parent);

	struct neat_genome *child;
	float random = (float)rand() / (float)RAND_MAX;
	if(random < p->conf.species_crossover_probability){
		/* Do a crossover with 2 parents if parent2 is valid */
		struct neat_genome *parent2 = neat_crossover_get_parent2(p, s);
		if(parent2){
			child = neat_genome_reproduce(parent, parent2);
		}else{
			child = neat_genome_copy(parent);
		}
	}else{
		/* Else copy the first parent */
		child = neat_genome_copy(parent);
	}

	random = (float)rand() / (float)RAND_MAX;
	if(random < p->conf.mutate_species_crossover_probability){
		neat_genome_mutate(child, p->conf, p->innovation);
	}

	neat_replace_genome(p, worst_genome, child);
	neat_genome_destroy(child);
}

static void neat_reproduce(struct neat_pop *p,
			   size_t worst_genome)
{
	assert(p);

	float total_avg = neat_get_species_fitness_average(p);

	float selection_random = (float)rand() / (float)RAND_MAX;
	for(size_t i = 0; i < p->nspecies; i++){
		struct neat_species *s = p->species[i];
		assert(s->ngenomes > 0);

		float avg = neat_species_get_average_fitness(s);
		float selection_prob = avg / total_avg;

		/* If we didn't find a match, 
		 * reduce the chance to find a new one
		 */
		if(selection_random > selection_prob){
			selection_random -= selection_prob;
			continue;
		}

		/* Select a random genome from the species, this will be the
		 * replacement if there is no crossover and a parent when 
		 * there is
		 */
		struct neat_genome *genitor = neat_species_select_genitor(s);
		/* Continue with finding proper species if the genitor could
		 * not be found
		 */
		if(!genitor){
			continue;
		}

		neat_crossover(p, s, worst_genome, genitor);

		neat_speciate_genome(p, worst_genome);

		break;
	}
}

neat_t neat_create(struct neat_config config)
{
	assert(config.population_size > 0);

	struct neat_pop *p = calloc(1, sizeof(struct neat_pop));
	assert(p);

	p->solved = false;
	p->conf = config;
	p->innovation = 1;

	/* Create a genome and copy it n times where n is the population size */
	p->ngenomes = config.population_size;
	p->genomes = malloc(sizeof(struct neat_genome*) *
			    config.population_size);
	assert(p->ngenomes);

	neat_reset_genomes(p);

	/* Create the starting species */
	p->nspecies = 0;
	p->species = NULL;
	neat_create_new_species(p, true);

	return p;
}

void neat_destroy(neat_t population)
{
	struct neat_pop *p = population;
	assert(p);

	for(size_t i = 0; i < p->ngenomes; i++){
		neat_genome_destroy(p->genomes[i]);
	}
	free(p->genomes);

	for(size_t i = 0; i < p->nspecies; i++){
		neat_species_destroy(p->species[i]);
	}
	free(p->species);
	free(p);
}

const float *neat_run(neat_t population,
		      size_t genome_id,
		      const float *inputs)
{
	struct neat_pop *p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	return neat_genome_run(p->genomes[genome_id], inputs);
}

bool neat_epoch(neat_t population, size_t *worst_genome)
{
	struct neat_pop *p = population;
	assert(p);

	p->innovation++;

	size_t worst_found_genome = 0;
	if(!neat_find_worst_fitness(p, &worst_found_genome)){
		return false;
	}

	/* Remove the worst genome from the species if it contains it */
	for(size_t i = 0; i < p->nspecies; i++){
		neat_species_remove_genome(p->species[i],
					   p->genomes[worst_found_genome]);
	}

	neat_reproduce(p, worst_found_genome);

	if(worst_genome){
		*worst_genome = worst_found_genome;
	}

	return true;
}

void neat_set_fitness(neat_t population, size_t genome_id, float fitness)
{
	struct neat_pop *p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	p->genomes[genome_id]->fitness = fitness;
}

void neat_increase_time_alive(neat_t population, size_t genome_id)
{
	struct neat_pop *p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	p->genomes[genome_id]->time_alive++;
}

const struct nn_ffnet *neat_get_network(neat_t population, size_t genome_id)
{
	struct neat_pop *p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	return p->genomes[genome_id]->net;
}

size_t neat_get_species_id(neat_t population, size_t genome_id)
{
	struct neat_pop *p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	struct neat_genome *genome = p->genomes[genome_id];

	for(size_t i = 0; i < p->nspecies; i++){
		if(neat_species_contains_genome(p->species[i], genome)){
			return i;
		}
	}

	return 0;
}

void neat_print_net(neat_t population, size_t genome_id)
{
	struct neat_pop *p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	neat_genome_print_net(p->genomes[genome_id]);
}
