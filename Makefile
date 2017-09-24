NAME=neat

RM=rm -rf
CFLAGS=-g -Wall -pedantic -O3 -Iinclude `pkg-config --cflags gtk+-3.0`
LDLIBS=-fopenmp `pkg-config --libs gtk+-3.0`

SRCS=tests/test.c src/network.c src/population.c src/species.c
OBJS=$(subst .c,.o,$(SRCS))

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(LDFLAGS) -o $(NAME) $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
