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

struct bird{
	/* Height and time since last "key press" */
	int last_height, last_time;
};

static int get_bird_height(struct bird b)
{
	float velocity_since_last, gravity_since_last;
	int time_squared;

	velocity_since_last = UP_KEY_VELOCITY * b.last_time;

	time_squared = b.last_time * b.last_time;

	gravity_since_last = GRAVITY * 0.5f * time_squared;

	return b.last_height + (int)(velocity_since_last + gravity_since_last);
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

	run = true;
	while(run){
		int keychar;

		keychar = getch();
		switch(keychar){
			case 'q':
				/* Exit the simulation */
				run = false;
		}

		clear();

		usleep(1000000u / FPS);
	}

	/* Close ncurses */
	endwin();

	return 0;
}
