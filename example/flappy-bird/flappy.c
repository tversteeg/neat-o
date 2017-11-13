#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

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

#define PIPE_DISTANCE 20

#define POPULATION 100

#define PIPES 100

struct pipe{
	/* The x position and the y position of the center of the opening */
	int x, center;

	/* Size of the opening */
	float opening_height;
};

struct bird{
	/* Height and time since last "key press" */
	int last_height, last_time;

	/* The x position */
	int x;
};

static struct bird birds[POPULATION];
static struct pipe pipes[PIPES];

static struct neat_config config;
static neat_t neat;

static int cam_x, best_distance;

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

static void draw_pipes(int middle)
{
	size_t i;

	for(i = 0; i < PIPES; i++){
		struct pipe p;
		int real_x, height;

		p = pipes[i];
		
		real_x = p.x - middle;
		/* Ignore the pipe if it's not in the screen */
		if(real_x < 1 || real_x >= WIDTH - 1){
			continue;
		}

		/* Draw top part of the pipe */
		height = HEIGHT - p.center - HEIGHT * p.opening_height;
    		mvvline(1, real_x, 0, height - 2);

		/* Draw bottom part of the pipe */
		height = HEIGHT - p.center + HEIGHT * p.opening_height;
    		mvvline(height, real_x, 0, (HEIGHT - 2) - height);
	}
}

static void draw_birds(void)
{
	size_t i;
	int middle;
	struct bird farthest;

	farthest = get_farthest_bird();
	if(farthest.x > cam_x){
		cam_x = farthest.x;
	}else{
		cam_x--;
	}
	middle = cam_x - WIDTH / 2;

	draw_pipes(middle);

	/* Draw the ground */
	mvhline(HEIGHT - 2, 1, 0, WIDTH - 2);

	/* Draw a moving figure to get a sense of speed */
    	mvaddch(HEIGHT - 2, (WIDTH - 1) - cam_x % (WIDTH - 2), ACS_BTEE);

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

	mvprintw(HEIGHT + 1, 1, "Current: %d", farthest.x);

	if(farthest.x > best_distance){
		best_distance = farthest.x;
	}

	mvprintw(HEIGHT + 2, 1, "Best: %d", best_distance);
}

static void reset_bird(struct bird *b)
{
	assert(b);

	b->last_height = HEIGHT / 2;
	b->last_time = 0;
	b->x = 0;
}

static struct pipe get_next_pipe(int x)
{
	size_t i;

	for(i = 0; i < PIPES; i++){
		if(pipes[i].x >= x){
			return pipes[i];
		}
	}

	/* TODO properly handle the case where there is no next pipe */
	return pipes[0];
}

static void run_network_on_bird(struct bird *b, size_t index)
{
	const float *results;
	struct pipe next;
	float fitness;
	float inputs[4];
	int height;

	assert(b);

	height = get_bird_height(*b);

	/* Get the nearest pipe */
	next = get_next_pipe(b->x);

	/* Get the difference in height between the middle of the pipe and the
	 * bird
	 */
	inputs[0] = next.center - height / (float)HEIGHT;

	/* Get the size of the pipe opening */
	inputs[1] = next.opening_height;

	/* Get the height itself */
	inputs[2] = height / (float)HEIGHT;

	/* Get the distance to the pipe */
	inputs[3] = (next.x - b->x) / (float)PIPE_DISTANCE;

	/* Run the bird network */
	results = neat_run(neat, index, inputs);
	assert(results);

	/* Make the bird jump if the output neuron is triggered */
	if(*results > 0.5f){
		b->last_height = height;
		b->last_time = 0;
	}

	/* The fitness is how far the bird flies */
	fitness = b->x / (float)(PIPE_DISTANCE * (PIPES + 1));
	neat_set_fitness(neat, index, fitness);
}

static bool bird_collides_with_pipe(struct bird b)
{
	size_t i;

	for(i = 0; i < PIPES; i++){
		struct pipe p;
		int bird_height;

		p = pipes[i];

		/* If the pipe and the bird are not on the same line there is
		 * no collision
		 */
		if(p.x != b.x){
			continue;
		}

		bird_height = get_bird_height(b);
		
		/* If the bird collides with top part */
		if(bird_height > p.center + HEIGHT * p.opening_height){
			return true;
		}

		/* If the bird collides with bottom part */
		if(bird_height < p.center - HEIGHT * p.opening_height){
			return true;
		}
	}

	/* No collision found */
	return false;
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

		/* Reset the bird if it collides with the pipes */
		if(bird_collides_with_pipe(*b)){
			reset_bird(b);
		}

		if(b->last_time > 10){
			run_network_on_bird(b, i);
		}

		neat_increase_time_alive(neat, i);
	}

	neat_epoch(neat, NULL);
}

static void initialize_pipes(void)
{
	size_t i;

	for(i = 0; i < PIPES; i++){
		int x;

		x = (i + 1) * PIPE_DISTANCE;

		pipes[i].x = x;

		/* Set the center between 1/4 and 3/4 of the height */
		pipes[i].center = HEIGHT / 4 + rand() % (HEIGHT / 2);
		
		/* TODO change this */
		pipes[i].opening_height = 0.25f - (i / 300.0f);
	}
}

static void initialize_population(void)
{
	size_t i;

	config = neat_get_default_config();
	config.network_inputs = 4;
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
	initialize_pipes();

	cam_x = 0;
	best_distance = 0;

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
