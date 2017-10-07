NAME=neat-test

RM=rm -rf
CFLAGS=-g -Wall -Werror -pedantic -O3 -Iinclude
LDLIBS=-fopenmp -lm

SRCS=test/test.c src/nn/nn.c src/neat/population.c
OBJS=$(SRCS:.c=.o)

TESTBINS=$(subst .c,,$(TESTS))

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
