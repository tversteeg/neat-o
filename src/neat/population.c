#include "population.h"

#include <float.h>
#include <assert.h>

static void neat_reset_genomes(struct neat_pop *p)
{
	size_t innovation;
	size_t i;

	assert(p);

	/* All the genomes will be random at start */
	innovation = p->innovation++;
	for(i = 0; i < p->ngenomes; i++){
		p->genomes[i] = neat_genome_create(p->conf, innovation);
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
	struct neat_species *new;

	assert(p);

	p->species = realloc(p->species,
			     sizeof(struct neat_species*) * ++p->nspecies);
	assert(p->species);

	new = neat_species_create(p->conf);

	if(fill){
		size_t i;
		for(i = 0; i < p->ngenomes; i++){
			neat_species_add_genome(new, i);
		}
	}

	p->species[p->nspecies - 1] = new;

	return new;
}

static void neat_remove_species_if_empty(struct neat_pop *p, size_t species_id)
{
	struct neat_species *s;

	assert(p);
	assert(species_id < p->nspecies);

	s = p->species[species_id];

	/* Only remove the species if there are no genomes left */
	if(s->ngenomes > 0){
		return;
	}

	neat_species_destroy(s);

	/* Put the last species on this position
	 * (this will do nothing if it already is the last one)
	 */
	p->species[species_id] = p->species[--p->nspecies];
	p->species[p->nspecies] = NULL;
}

static bool neat_find_worst_fitness(struct neat_pop *p, size_t *worst_genome)
{
	bool found_worst;
	float worst_fitness;
	size_t i;

	assert(p);
	assert(worst_genome);

	found_worst = false;

	worst_fitness = FLT_MAX;
	for(i = 0; i < p->ngenomes; i++){
		struct neat_genome *genome;
		float fitness;

		genome = p->genomes[i];

		fitness = genome->fitness;
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
	float total_avg;
	size_t i;

	assert(p);

	total_avg = 0.0;
	for(i = 0; i < p->nspecies; i++){
		total_avg += neat_species_get_average_fitness(p, p->species[i]);
	}
	total_avg /= (double)p->nspecies;

	return total_avg;
}

static void neat_speciate_genome(struct neat_pop *p, size_t genome_id)
{
	struct neat_genome *genome;
	struct neat_species *new;
	float compatibility_treshold;
	size_t i;

	assert(p);

	genome = p->genomes[genome_id];
	compatibility_treshold = p->conf.genome_compatibility_treshold;

	/* Add genome to species if the representant matches the genome */
	for(i = 0; i < p->nspecies; i++){
		size_t id;
		struct neat_genome *species_representant;

		if(p->species[i]->ngenomes == 0){
			continue;
		}
		id = neat_species_get_representant(p->species[i]);
		species_representant = p->genomes[id];
		if(neat_genome_is_compatible(genome,
					     species_representant,
					     compatibility_treshold)){
			neat_species_add_genome(p->species[i], genome_id);
			return;
		}
	}

	/* If no matching species could be found create a new species */
	new = neat_create_new_species(p, false);
	neat_species_add_genome(new, genome_id);
}

static struct neat_genome *neat_crossover_get_parent2(struct neat_pop *p,
						      struct neat_species *s)
{
	float random;

	assert(p);
	assert(s);

	random = (float)rand() / (float)RAND_MAX;

	/* Check for interspecies mutation which won't work with
	 * only 1 species
	 */
	if(p->nspecies > 1 &&
	   random < p->conf.interspecies_crossover_probability){
		size_t species_index;
		struct neat_species *random_species;

		species_index = rand() % p->nspecies;
		random_species = p->species[species_index];
		assert(random_species);

		/* We don't want to do interspecies crossover on the same
		 * species
		 */
		if(random_species == s){
			/* We can't move the index lower than 0 */
			if(species_index == 0){
				random_species = p->species[species_index + 1];
			}else{
				random_species = p->species[species_index - 1];
			}
		}

		if(random_species->ngenomes == 0){
			return NULL;
		}

		return p->genomes[neat_species_select_genitor(random_species)];
	}else{
		return p->genomes[neat_species_select_genitor(s)];
	}
}

static void neat_crossover(struct neat_pop *p,
			   struct neat_species *s,
			   size_t worst_genome,
			   struct neat_genome *parent)
{
	struct neat_genome *child;
	float random;

	assert(p);
	assert(s);
	assert(parent);

	random = (float)rand() / (float)RAND_MAX;
	if(random < p->conf.species_crossover_probability){
		struct neat_genome *parent2;

		/* Do a crossover with 2 parents if parent2 is valid */
		parent2 = neat_crossover_get_parent2(p, s);
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

	/* Reset the time alive for the child */
	child->time_alive = 0;

	neat_replace_genome(p, worst_genome, child);
	neat_genome_destroy(child);
}

static void neat_reproduce(struct neat_pop *p,
			   size_t worst_genome)
{
	float total_avg, selection_random;
	size_t i;

	assert(p);

	total_avg = neat_get_species_fitness_average(p);

	selection_random = (float)rand() / (float)RAND_MAX;
	for(i = 0; i < p->nspecies; i++){
		struct neat_species *s;
		struct neat_genome *genitor;
		float avg, selection_prob;
		size_t genitor_id;

		s = p->species[i];
		assert(s->ngenomes > 0);

		avg = neat_species_get_average_fitness(p, s);
		selection_prob = avg / total_avg;

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
		genitor_id = neat_species_select_genitor(s);
		genitor = p->genomes[genitor_id];
		/* Continue with finding proper species if the genitor could
		 * not be found
		 */
		if(!genitor){
			continue;
		}

		neat_crossover(p, s, worst_genome, genitor);

		if(p->conf.speciate){
			neat_speciate_genome(p, worst_genome);
		}

		break;
	}
}

neat_t neat_create(struct neat_config config)
{
	struct neat_pop *p;

	assert(config.network_inputs > 0);
	assert(config.network_outputs > 0);
	assert(config.network_hidden_nodes > 0);
	assert(config.population_size > 0);
	assert(config.minimum_time_before_replacement > 0);

	p = calloc(1, sizeof(struct neat_pop));
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
	struct neat_pop *p;
	size_t i;

	p = population;
	assert(p);

	for(i = 0; i < p->ngenomes; i++){
		neat_genome_destroy(p->genomes[i]);
	}
	free(p->genomes);

	for(i = 0; i < p->nspecies; i++){
		neat_species_destroy(p->species[i]);
	}
	free(p->species);
	free(p);
}

const float *neat_run(neat_t population,
		      size_t genome_id,
		      const float *inputs)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	return neat_genome_run(p->genomes[genome_id], inputs);
}

bool neat_epoch(neat_t population, size_t *worst_genome)
{
	struct neat_pop *p;
	size_t i, worst_found_genome;

	p = population;
	assert(p);

	p->ticks++;

	/* Wait for the set amount of ticks until a replacement occurs */
	if(p->ticks % p->conf.minimum_time_before_replacement != 0){
		return false;
	}

	p->innovation++;

	worst_found_genome = 0;
	if(!neat_find_worst_fitness(p, &worst_found_genome)){
		return false;
	}

	/* Remove the worst genome from the species if it contains it */
	for(i = 0; i < p->nspecies; i++){
		if(neat_species_remove_genome_if_exists(p->species[i],
							worst_found_genome)){
			neat_remove_species_if_empty(p, i);
		}
	}

	neat_reproduce(p, worst_found_genome);

	if(worst_genome){
		*worst_genome = worst_found_genome;
	}

	return true;
}

void neat_set_fitness(neat_t population, size_t genome_id, float fitness)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	p->genomes[genome_id]->fitness = fitness;
}

void neat_increase_time_alive(neat_t population, size_t genome_id)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	p->genomes[genome_id]->time_alive++;
}

const struct nn_ffnet *neat_get_network(neat_t population, size_t genome_id)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	return p->genomes[genome_id]->net;
}

size_t neat_get_species_id(neat_t population, size_t genome_id)
{
	struct neat_pop *p;
	size_t i;

	p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	for(i = 0; i < p->nspecies; i++){
		if(neat_species_contains_genome(p->species[i], genome_id)){
			return i;
		}
	}

	return 0;
}

void neat_print_net(neat_t population, size_t genome_id)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	neat_genome_print_net(p->genomes[genome_id]);
}
