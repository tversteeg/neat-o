#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <unistd.h>
#include <ncurses.h>

#include <nn.h>
#include <neat.h>

/* Frames per second */
#define FPS 24u

#define GRAVITY 0.05f
#define UP_KEY_VELOCITY -0.5f
#define HEIGHT 80

#define POPULATION 100

struct bird{
	/* Height and time since last "key press" */
	int last_height, last_time;

	/* The x position */
	int x;
};

static struct bird birds[POPULATION];

static struct neat_config config;
static neat_t neat;

static int current_time;

static int get_bird_height(struct bird b)
{
	float velocity_since_last, gravity_since_last;
	int time_squared;

	velocity_since_last = UP_KEY_VELOCITY * b.last_time;

	time_squared = b.last_time * b.last_time;

	gravity_since_last = GRAVITY * 0.5f * time_squared;

	return b.last_height + (int)(velocity_since_last + gravity_since_last);
}

static void population_tick(void)
{
	size_t i;

	for(i = 0; i < POPULATION; i++){
		const float *results;
		int height;
		struct bird *b;
		float fitness, dist_to_middle;

		b = birds + i;

		height = get_bird_height(*b);

		/* Run the bird network */
		results = neat_run(neat, i, &height);
		assert(results);

		/* Make the bird jump if the output neuron is triggered */
		if(*results > 0.5f){
			b->last_height = height;
			b->last_time = current_time;
		}

		/* We want the bird to stay as close to the middle as possible
		 * but also to get as far as possible
		 */
		dist_to_middle = (height - HEIGHT / 2) / (float)HEIGHT;
		fitness = dist_to_middle * 0.1f + (b->x / 100.0f)
		neat_set_fitness(neat, i, fitness);

		neat_increase_time_alive(neat, i);
	}

	neat_epoch(neat, NULL);

	current_time++;
}

static void initialize_population(void)
{
	size_t i;

	config = neat_get_default_config();
	config.network_inputs = 1;
	config.network_outputs = 1;
	config.network_hidden_nodes = 6;
	config.population_size = POPULATION;

	neat = neat_create(config);
	assert(neat);

	for(i = 0; i < POPULATION; i++){
		struct bird *b;

		b = birds + i;
		b->last_time = 0;
		b->last_height = 0;
		b->x = 0;
	}
}

static void initialize_ncurses(void)
{
	initscr();
	raw();
	keypad(stdscr, TRUE);
	noecho();
	curs_set(0);
	timeout(0);
}

int main(int argc, char *argv[])
{
	bool run;

	(void)argc;
	(void)argv;

	srand(time(NULL));

	initialize_ncurses();

	initialize_population();

	current_time = 0;

	run = true;
	while(run){
		/* Get the key pressed */
		switch(getch()){
			case 'q':
				/* Exit the simulation */
				run = false;
		}

		clear();

		population_tick();

		usleep(1000000u / FPS);
	}

	/* Close ncurses */
	endwin();

	return 0;
}
