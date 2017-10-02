NAME=neat-test

RM=rm -rf
CFLAGS=-g -Wall -Werror -pedantic -O3 -Iinclude
LDLIBS=-fopenmp

SRCS=test/test.c src/nn/nn.c
OBJS=$(SRCS:.c=.o)

TESTBINS=$(subst .c,,$(TESTS))

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
