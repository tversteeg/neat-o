NAME=neat

TESTS=tests/nn.c tests/neat.c

RM=rm -rf
CFLAGS=-g -Wall -pedantic -O3 -Iinclude
LDLIBS=-fopenmp

SRCS=src/neat/population.c src/neat/species.c \
     src/nn/nn.c \
     $(TESTS)
OBJS=$(subst .c,.o,$(SRCS))

TESTBINS=$(subst .c,,$(TESTS))

all: $(TESTBINS)

$(TESTBINS): $(OBJS)
	$(CC) $(LDFLAGS) -o $(NAME) $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
