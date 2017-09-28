NAME=neat

RM=rm -rf
CFLAGS=-g -Wall -pedantic -O3 -Iinclude
LDLIBS=-fopenmp

SRCS=src/network.c src/population.c src/species.c src/gene.c src/neuron.c \
     tests/test.c 

OBJS=$(subst .c,.o,$(SRCS))

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(LDFLAGS) -o $(NAME) $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
