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

#define WIDTH 80
#define HEIGHT 24

#define GRAVITY 0.05f
#define UP_KEY_VELOCITY -0.5f

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

static int get_bird_height(struct bird b)
{
	float velocity_since_last, gravity_since_last, time_squared;

	velocity_since_last = UP_KEY_VELOCITY * (float)b.last_time;

	time_squared = b.last_time * b.last_time;

	gravity_since_last = GRAVITY * 0.5f * time_squared;

	return b.last_height + (int)(velocity_since_last + gravity_since_last);
}

static struct bird get_farthest_bird(void)
{
	size_t i;
	int farthest = 0;
	struct bird farthest_bird;

	for(i = 0; i < POPULATION; i++){
		struct bird b;

		b = birds[i];
		if(b.x > farthest){
			farthest = b.x;
			farthest_bird = b;
		}
	}
	
	return farthest_bird;
}

static void draw_birds(void)
{
	size_t i;
	int middle;
	struct bird farthest;

	farthest = get_farthest_bird();
	middle = farthest.x - WIDTH / 2;

	/* Draw the ground */
	mvhline(HEIGHT - 2, 1, 0, WIDTH - 2);

	/* Draw a moving figure to get a sense of speed */
    	mvaddch(HEIGHT - 2, (WIDTH - 1) - farthest.x % (WIDTH - 2), ACS_BTEE);

	for(i = 0; i < POPULATION; i++){
		struct bird *b;
		int real_x, real_y;

		b = birds + i;
		
		real_x = b->x - middle;
		real_y = HEIGHT - get_bird_height(*b) - 2;

		/* Ignore the bird if it's not in the screen */
		if(real_x < 1 || real_x >= WIDTH - 1){
			continue;
		}
		if(real_y < 1 || real_y >= HEIGHT - 1){
			continue;
		}

		mvprintw(real_y, real_x, "o");
	}

	mvprintw(HEIGHT + 1, 1, "%d", farthest.x);
}

static void reset_bird(struct bird *b)
{
	assert(b);

	b->last_height = HEIGHT / 2;
	b->last_time = 0;
	b->x = 0;
}

static void run_network_on_bird(struct bird *b, size_t index)
{
	const float *results;
	float height_norm, dist_to_middle, fitness;
	int height;

	assert(b);

	height = get_bird_height(*b);

	/* Run the bird network */
	height_norm = height / (float)HEIGHT;
	results = neat_run(neat, index, &height_norm);
	assert(results);

	/* Make the bird jump if the output neuron is triggered */
	if(*results > 0.5f){
		b->last_height = height;
		b->last_time = 0;
	}

	/* We want the bird to stay as close to the middle as possible but also
	 * to get as far as possible
	 */
	dist_to_middle = (height - HEIGHT / 2) / (float)HEIGHT;
	fitness = dist_to_middle * 0.1f + (b->x / 100.0f);
	neat_set_fitness(neat, index, fitness);
}

static void population_tick(void)
{
	size_t i;

	for(i = 0; i < POPULATION; i++){
		int height;
		struct bird *b;

		b = birds + i;

		b->x++;
		b->last_time++;
		height = get_bird_height(*b);

		/* Reset the bird if it's out of bounds */
		if(height > HEIGHT || height < 0){
			reset_bird(b);
		}

		if(b->last_time > 10){
			run_network_on_bird(b, i);
		}

		neat_increase_time_alive(neat, i);
	}

	neat_epoch(neat, NULL);
}

static void initialize_population(void)
{
	size_t i;

	config = neat_get_default_config();
	config.network_inputs = 1;
	config.network_outputs = 1;
	config.network_hidden_nodes = 6;
	config.population_size = POPULATION;

	config.genome_minimum_ticks_alive = 50;
	config.minimum_time_before_replacement = 200;

	neat = neat_create(config);
	assert(neat);

	for(i = 0; i < POPULATION; i++){
		reset_bird(birds + i);
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

static void draw_ncurses_rectangle(int y1, int x1, int y2, int x2)
{
    mvhline(y1, x1, 0, x2-x1);
    mvhline(y2, x1, 0, x2-x1);
    mvvline(y1, x1, 0, y2-y1);
    mvvline(y1, x2, 0, y2-y1);
    mvaddch(y1, x1, ACS_ULCORNER);
    mvaddch(y2, x1, ACS_LLCORNER);
    mvaddch(y1, x2, ACS_URCORNER);
    mvaddch(y2, x2, ACS_LRCORNER);
}

int main(int argc, char *argv[])
{
	bool run;

	(void)argc;
	(void)argv;

	srand(time(NULL));

	initialize_ncurses();

	initialize_population();

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

		draw_ncurses_rectangle(0, 0, HEIGHT, WIDTH);
		draw_birds();

		refresh();

		usleep(1000000u / FPS);
	}

	/* Close ncurses */
	endwin();

	return 0;
}
