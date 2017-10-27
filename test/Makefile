NAME=neat-test

RM=rm -rf
CFLAGS=-g -Wall -Werror -pedantic -O3 -I../include
LDLIBS=-fopenmp -lm

SRCS=test.c ../src/nn/nn.c \
     ../src/neat/population.c ../src/neat/species.c ../src/neat/genome.c
OBJS=$(SRCS:.c=.o)

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

.PHONY: test
test: $(NAME)
	$(NAME)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
