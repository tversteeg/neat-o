#include "population.h"

#include <stdint.h>
#include <float.h>
#include <assert.h>

static size_t neat_random_eligible_species_list(struct neat_pop *p,
						size_t **eligible_list)
{
	size_t i, eligible_count, *list;

	assert(p);
	assert(eligible_list);

	*eligible_list = malloc(sizeof(size_t) * p->nspecies);
	assert(*eligible_list);

	list = *eligible_list;

	/* First fill the array if the species are eligible */
	eligible_count = 0;
	for(i = 0; i < p->nspecies; i++){
		if(p->species[i]->active){
			list[eligible_count++] = i;
		}
	}

	/* If there are no eligible species create a new one */
	if(eligible_count == 0){
		free(list);
		*eligible_list = NULL;

		return 0;
	}

	/* Then shuffle it randomly */
	for(i = 0; i < eligible_count - 1; i++){
		size_t j, tmp;

		/* Select a next random item starting from the current
		 * position
		 */
		j = i + rand() / (RAND_MAX / (eligible_count - i) + 1);

		/* Swap the selected zitem with the current one */
		tmp = list[j];
		list[j] = list[i];
		list[i] = tmp;
	}

	/* Resize the list to remove unneeded items */
	*eligible_list = realloc(list, sizeof(size_t) * eligible_count);

	return eligible_count;
}

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

static void neat_remove_genome_from_species(struct neat_pop *p,
					    size_t genome_id)
{
	size_t i;

	for(i = 0; i < p->nspecies; i++){
		if(neat_species_remove_genome_if_exists(p->species[i],
							genome_id)){
			neat_remove_species_if_empty(p, i);
			/* Move i one back because the array has been
			 * changed by the possible removal of the species
			 */
			i--;
		}
	}
}

static float neat_adjusted_genome_fitness(struct neat_pop *p, size_t genome_id)
{
	float fitness;
	size_t i;

	assert(p);

	fitness = p->genomes[genome_id]->fitness;

	for(i = 0; i < p->nspecies; i++){
		if(neat_species_contains_genome(p->species[i], genome_id)){
			return neat_species_get_adjusted_fitness(p->species[i],
								 fitness);

		}
	}

	/* All genomes should be inside species */
	assert(false);

	return 0.0f;
}

static int neat_sort_species_compare(const void *a, const void *b)
{
	float s1_fitness, s2_fitness;
	struct neat_species *s1, *s2;

	s1 = *(struct neat_species**)a;
	s2 = *(struct neat_species**)b;

	assert(s1);
	assert(s2);

	s1_fitness = s1->avg_fitness;
	s2_fitness = s2->avg_fitness;

	/* Sort from high to low */
	if(s1_fitness < s2_fitness){
		return 1;
	}else if(s1_fitness > s2_fitness){
		return -1;
	}else{
		return 0;
	}
}

static bool neat_find_worst_genome(struct neat_pop *p, size_t *worst_genome)
{
	bool found_worst;
	float worst_fitness;
	size_t i;

	assert(p);
	assert(worst_genome);

	found_worst = false;

	/* First look if there are still dead species */
	for(i = 0; i < p->nspecies; i++){
		struct neat_species *species;

		species = p->species[i];

		if(!species->active){
			/* Just select the first genome in the species,
			 * because when it will be removed the array will be
			 * automatically slided left
			 */
			*worst_genome = species->genomes[0];
			return true;
		}
	}

	/* Finally check the rest, we iterate back so the same genome won't be
	 * chosen when there are only dead species
	 */
	worst_fitness = FLT_MAX;
	for(i = p->ngenomes; i > 0; i--){
		struct neat_genome *genome;
		float fitness;

		genome = p->genomes[i - 1];

		/* Find the genome with the lowest adjusted fitness */
		fitness = neat_adjusted_genome_fitness(p, i - 1);
		if(fitness < worst_fitness &&
		   genome->time_alive > p->conf.genome_minimum_ticks_alive){
			*worst_genome = i - 1;
			worst_fitness = fitness;
			found_worst = true;
		}
	}

	return found_worst;
}

static float neat_get_total_fitness_average(struct neat_pop *p)
{
	float total_avg;
	size_t i;

	assert(p);

	total_avg = 0.0;
	for(i = 0; i < p->nspecies; i++){
		total_avg += p->species[i]->avg_fitness;
	}
	total_avg /= (double)p->nspecies;

	return total_avg;
}

static void neat_increase_all_species_generations(struct neat_pop *p)
{
	size_t i;

	assert(p);

	for(i = 0; i < p->nspecies; i++){
		neat_species_increase_generation(p->species[i]);
	}
}

static void neat_update_all_species_averages(struct neat_pop *p)
{
	size_t i;

	assert(p);

	for(i = 0; i < p->nspecies; i++){
		neat_species_update_average_fitness(p, p->species[i]);
	}
}

static void neat_cull_species(struct neat_pop *p)
{
	size_t i;

	assert(p);

	/* We walk backward so the iterator won't be ruined when species
	 * will be removed
	 */
	for(i = p->nspecies; i > 0; i--){
		struct neat_species *species;

		species = p->species[i - 1];
		assert(species);

		neat_species_cull(p, species);
	}
}

static void neat_remove_duplicate_species(struct neat_pop *p)
{
	size_t i, compatibility_treshold;

	assert(p);

	compatibility_treshold = p->conf.genome_compatibility_treshold;

	for(i = 0; i < p->nspecies; i++){
		struct neat_species *s1;
		size_t j;

		s1 = p->species[i];

		/* Ignore all the dead species */
		if(!s1->active){
			continue;
		}

		for(j = i + 1; j < p->nspecies; j++){
			struct neat_species *s2;
			struct neat_genome *r1, *r2;

			s2 = p->species[j];
			
			/* Ignore the dead species */
			if(!s2->active){
				continue;
			}

			/* Remove all the species where the representants are 
			 * compatible
			 */
			/* TODO make this more safe */
			r1 = p->genomes[neat_species_get_representant(s1)];
			r2 = p->genomes[neat_species_get_representant(s2)];

			if(!neat_genome_is_compatible(r1,
						     r2,
						     compatibility_treshold,
						     p->nspecies)){
				continue;
			}

			/* Keep the species with the highest fitness */
			if(s1->avg_fitness > s2->avg_fitness){
				s2->active = false;
			}else{
				s1->active = false;
				break;
			}
		}
	}
}

static bool neat_add_genome_to_eligible_species(struct neat_pop *p,
						size_t genome_id)
{
	struct neat_genome *genome;
	float compatibility_treshold;
	size_t i, *eligible_species, eligible_count;

	assert(p);

	/* Create a list of random id's where we can loop through, this
	 * makes sure the species are not selected in the same order and thus
	 * the same species won't fill up if they are compatible
	 */
	eligible_species = NULL;
	eligible_count = neat_random_eligible_species_list(p,
							   &eligible_species);
	if(eligible_count == 0){
		return false;
	}
	assert(eligible_species);

	genome = p->genomes[genome_id];
	compatibility_treshold = p->conf.genome_compatibility_treshold;

	/* Add genome to species if the representant matches the genome */
	for(i = 0; i < eligible_count; i++){
		size_t j, rep_id;
		struct neat_genome *species_representant;

		j = eligible_species[i];

		/* All empty species should be removed at this point */
		assert(p->species[j]->ngenomes > 0);

		rep_id = neat_species_get_representant(p->species[j]);
		species_representant = p->genomes[rep_id];

		/* Add the genome to the species if it's compatible,
		 * compatibility is checked with a treshold which is again
		 * influenced by the number of species that exist, more
		 * species means a bigger chance of compatibility which means
		 * less new species (and the other way around as well)
		 */
		if(neat_genome_is_compatible(genome,
					     species_representant,
					     compatibility_treshold,
					     p->nspecies)){
			neat_species_add_genome(p->species[j], genome_id);

			/* Cleanup */
			free(eligible_species);
			return true;
		}
	}

	/* No compatible species found */
	free(eligible_species);
	return false;
}

static void neat_speciate_genome(struct neat_pop *p, size_t genome_id)
{
	struct neat_species *new;

	assert(p);
	assert(p->nspecies > 0);

	if(!neat_add_genome_to_eligible_species(p, genome_id)){
		/* If no matching eligible species could be found create a
		 * new species
		 */
		new = neat_create_new_species(p, false);
		neat_species_add_genome(new, genome_id);
	}
}

static void neat_respeciate_genomes(struct neat_pop *p)
{
	size_t i;

	assert(p);

	for(i = 0; i < p->ngenomes; i++){
		/* First remove the species, this will have the side effect
		 * that representants will be shifted in species
		 * TODO check if that's a problem
		 */
		neat_remove_genome_from_species(p, i);

		/* Then assign it to a existing species or create a new species
		 */
		neat_speciate_genome(p, i);
	}
}

static struct neat_species *neat_interspecies_species(struct neat_pop *p,
						      struct neat_species *s)
{
	size_t i, eligible_count, *eligible_species;

	eligible_species = NULL;
	eligible_count = neat_random_eligible_species_list(p,
							   &eligible_species);
	if(eligible_count == 0){
		return NULL;
	}
	assert(eligible_species);

	for(i = 0; i < eligible_count; i++){
		struct neat_species *s2 = p->species[eligible_species[i]];
		assert(s2);

		/* Parent2 cannot be the same as parent1 */
		if(s2 != s){
			return s2;
		}
	}

	/* No eligible parent found */
	return NULL;
}

static struct neat_genome *neat_crossover_get_parent2(struct neat_pop *p,
						      struct neat_species *s)
{
	float random;
	size_t genitor;
	struct neat_species *parent2_species;

	assert(p);
	assert(s);

	genitor = SIZE_MAX;

	random = (float)rand() / (float)RAND_MAX;
	/* TODO fix interspecies crossover not ignoring inactive species */
	if(random < p->conf.interspecies_crossover_probability && false){
		parent2_species = neat_interspecies_species(p, s);
		if(parent2_species != NULL){
			/* Get the best genome from the randomly selected
			 * species
			 */
			genitor = neat_species_select_genitor(p,
							      parent2_species);
		}
	}
	
	/* If there is no interspecies crossover or if no suitable species
	 * could be found
	 */
	if(genitor == SIZE_MAX){
		/* Get the second best one if there is only 1 species (so that
		 * would be the first) or if there is a non-interspecies
		 * crossover
		 */
		genitor = neat_species_select_second_genitor(p, s);
	}

	return p->genomes[genitor];
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

		if(parent == parent2){
			/* Just copy one of the parents if they are the same */
			child = neat_genome_copy(parent);
		}else{
			child = neat_genome_reproduce(parent, parent2);
		}
	}else{
		/* Else copy the first parent */
		child = neat_genome_copy(parent);
	}

	neat_genome_mutate(child, p->conf, p->innovation);

	/* Reset the time alive for the child */
	child->time_alive = 0;

	neat_replace_genome(p, worst_genome, child);
	neat_genome_destroy(child);
}

static void neat_reproduce(struct neat_pop *p, size_t worst_genome)
{
	float total_avg, selection_random;
	size_t i;

	assert(p);

	neat_update_all_species_averages(p);

	neat_cull_species(p);

	/* Sort species on fitness */
	qsort(p->species,
	      p->nspecies,
	      sizeof(struct neat_species*),
	      neat_sort_species_compare);

	total_avg = neat_get_total_fitness_average(p);

	selection_random = (float)rand() / (float)RAND_MAX;
	for(i = 0; i < p->nspecies; i++){
		struct neat_species *s;
		struct neat_genome *genitor;
		float selection_prob;
		size_t genitor_id;

		s = p->species[i];
		assert(s->ngenomes > 0);

		selection_prob = s->avg_fitness / total_avg;

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
		genitor_id = neat_species_select_genitor(p, s);
		assert(genitor_id < p->ngenomes);

		genitor = p->genomes[genitor_id];
		assert(genitor);

		/* Crossover replaces the worst genome with a new one */
		neat_crossover(p, s, worst_genome, genitor);

		/* And then assign it to one or create a new one if no species
		 * matches
		 */
		neat_speciate_genome(p, worst_genome);

		break;
	}
}

struct neat_config neat_get_default_config(void)
{
	/* TODO make it pretty, now it's not the most
	 * pretty way to initialize it
	 */
	struct neat_config conf = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0};

	conf.minimum_time_before_replacement = 10;

	conf.species_stagnation_treshold = 100;
	conf.species_stagnations_allowed = 2;
	conf.species_ticks_before_reassignment = 10;

	conf.species_crossover_probability = 0.6;
	conf.interspecies_crossover_probability = 0.2;

	/*
	conf.genome_add_neuron_mutation_probability = 0.1;
	conf.genome_add_link_mutation_probability = 0.3;
	conf.genome_change_activation_probability = 0.1;
	conf.genome_weight_mutation_probability = 0.5;
	conf.genome_all_weights_mutation_probability = 0.02;
	*/

	conf.genome_minimum_ticks_alive = 100;
	conf.genome_compatibility_treshold = 0.2;

	conf.genome_default_hidden_activation = NN_ACTIVATION_RELU;
	conf.genome_default_output_activation = NN_ACTIVATION_SIGMOID;

	return conf;
}

neat_t neat_create(struct neat_config config)
{
	struct neat_pop *p;

	assert(config.network_inputs > 0);
	assert(config.network_outputs > 0);
	assert(config.network_hidden_nodes > 0);
	assert(config.population_size > 0);
	assert(config.minimum_time_before_replacement > 0);

	/* Set all the allocated values to 0 so we don't have to manually
	 * set everything
	 */
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

const float *neat_run(neat_t population, size_t genome_id, const float *inputs)
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
	size_t worst_found_genome, nspecies, reassignment_ticks;
	bool respeciate;

	p = population;
	assert(p);

	/* Wait for the set amount of ticks until a replacement occurs */
	if(++p->ticks % p->conf.minimum_time_before_replacement != 0){
		return false;
	}

	/* Cache the variable to track the difference */
	nspecies = p->nspecies;

	p->innovation++;

	/* Increment the generation counter on each species so later it can
	 * be checked if it needs to be culled
	 */
	neat_increase_all_species_generations(p);

	/* Update all the averages so the worst genome can be found */
	neat_update_all_species_averages(p);

	/* Remove species where the fitness is exactly the same */
	neat_remove_duplicate_species(p);

	/* Reassign all the genomes if the compatibility treshold is
	 * adjusted (which is done by changing the amount of species)
	 */
	respeciate = false;
	if(nspecies != p->nspecies){
		/* Respeciate only with an interval defined in the config */
		reassignment_ticks = p->conf.species_ticks_before_reassignment;
		respeciate = reassignment_ticks >= p->reassignment_ticks++;
	}

	worst_found_genome = 0;
	if(!neat_find_worst_genome(p, &worst_found_genome)){
		if(respeciate){
			neat_respeciate_genomes(p);

			/* Reset respeciation timer */
			p->reassignment_ticks = 0;
		}

		return false;
	}

	/* Remove the worst genome from the species if it contains it
	 * and destroy the species if it's empty
	 */
	neat_remove_genome_from_species(p, worst_found_genome);

	/* Fill the now empty worst genome slot with a newly created one */
	neat_reproduce(p, worst_found_genome);

	if(worst_genome){
		*worst_genome = worst_found_genome;
	}

	/* Check the amount of species again because removal and 
	 * reproduction might have changed the amount of species
	 */
	if(nspecies != p->nspecies){
		/* Respeciate only with an interval defined in the config */
		reassignment_ticks = p->conf.species_ticks_before_reassignment;
		respeciate = reassignment_ticks >= p->reassignment_ticks;

		if(respeciate){
			neat_respeciate_genomes(p);

			/* Reset respeciation timer */
			p->reassignment_ticks = 0;
		}
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

	/* Return the first species that contains the genome id (there should
	 * only be one anyway)
	 */
	for(i = 0; i < p->nspecies; i++){
		if(neat_species_contains_genome(p->species[i], genome_id)){
			return i;
		}
	}

	return 0;
}

size_t neat_get_num_species(neat_t population)
{
	struct neat_pop *p;

	p = population;
	assert(p);

	return p->nspecies;
}

size_t neat_get_num_genomes_in_species(neat_t population, size_t species_id)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(species_id < p->nspecies);

	return p->species[species_id]->ngenomes;
}

float neat_get_average_fitness_of_species(neat_t population, size_t species_id)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(species_id < p->nspecies);

	return p->species[species_id]->avg_fitness;
}

bool neat_get_species_is_alive(neat_t population, size_t species_id)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(species_id < p->nspecies);

	return p->species[species_id]->active;
}

void neat_print_net(neat_t population, size_t genome_id)
{
	struct neat_pop *p;

	p = population;
	assert(p);
	assert(genome_id < p->ngenomes);

	neat_genome_print_net(p->genomes[genome_id]);
}
