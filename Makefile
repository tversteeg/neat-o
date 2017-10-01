NAME=neat

TESTS=test/nn.c test/neat.c

RM=rm -rf
CFLAGS=-g -Wall -pedantic -O3 -Iinclude
LDLIBS=-fopenmp

SRCS=src/neat/population.c src/neat/species.c \
     src/nn/nn.c
OBJS=$(SRCS:.c=.o)

TESTBINS=$(subst .c,,$(TESTS))

all: $(TESTBINS)

$(TESTBINS): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $@.o $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
