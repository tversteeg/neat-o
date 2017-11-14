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

static WINDOW *window;

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
    		mvwvline(window, 1, real_x, 0, height - 1);

		/* Draw bottom part of the pipe */
		height = HEIGHT - p.center + HEIGHT * p.opening_height;
    		mvwvline(window, height, real_x, 0, HEIGHT - height);
	}
}

static void draw_birds(void)
{
	size_t i;
	int middle, farthest_pipes;
	struct bird farthest;

	farthest = get_farthest_bird();
	if(farthest.x > cam_x){
		cam_x = farthest.x;
	}else if(cam_x - farthest.x > WIDTH / 2){
		cam_x--;
	}
	middle = cam_x - WIDTH / 2;

	for(i = 0; i < POPULATION; i++){
		struct bird *b;
		int real_x, real_y, height;

		b = birds + i;

		height = get_bird_height(*b);
		
		real_x = b->x - middle;
		real_y = HEIGHT - height - 1;

		/* Ignore the bird if it's not in the screen */
		if(real_x < 1 || real_x >= WIDTH - 1){
			continue;
		}
		if(real_y < 1 || real_y >= HEIGHT - 1){
			continue;
		}

		/* Draw a bird depending on if it just flapped */
		if(b->last_time < 10 && (b->last_time / 4) % 2 == 0){
			if(real_x > 1){
				mvwprintw(window, real_y, real_x - 1, "-");
			}
			mvwprintw(window, real_y, real_x, "o-");
		}else{
			if(real_x > 1){
				mvwprintw(window, real_y, real_x - 1, "\\");
			}
			mvwprintw(window, real_y, real_x, "o/");
		}
	}

	draw_pipes(middle);

	farthest_pipes = farthest.x / (PIPE_DISTANCE - 1);
	mvprintw(HEIGHT + 1, 1, "Current: %d", farthest_pipes);

	if(farthest_pipes > best_distance){
		best_distance = farthest_pipes;
	}

	mvprintw(HEIGHT + 2, 1, "Best: %d", best_distance);
}

static void reset_bird(struct bird *b)
{
	assert(b);

	/* Make the height start between 1/4 and 3/4 of the screen */
	b->last_height = HEIGHT / 4 + rand() % (HEIGHT / 2);
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
	config.network_hidden_nodes = 10;
	config.population_size = POPULATION;

	config.genome_minimum_ticks_alive = 200;
	config.minimum_time_before_replacement = 50;

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

	window = newwin(HEIGHT, WIDTH, 0, 0);
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

		population_tick();

		/* Clear the window area */
		werase(window);

		draw_birds();
		
		/* Draw the window edges */
		box(window, 0, 0);

		/* Draw everything */
		wrefresh(window);
		refresh();

		usleep(1000000u / FPS);
	}

	/* Close ncurses */
	delwin(window);

	endwin();

	return 0;
}
