NAME=neat

RM=rm -rf
CFLAGS=-g -Wall -pedantic -O3 -Iinclude
LDLIBS=-fopenmp

SRCS=tests/test.c src/network.c src/population.c src/genome.c
OBJS=$(subst .c,.o,$(SRCS))

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(LDFLAGS) -o $(NAME) $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
