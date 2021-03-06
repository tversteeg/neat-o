LIB=neat.a

AR=ar rcs
RANLIB=ranlib
CFLAGS=-g -Wall -Wextra -Wmissing-prototypes -Wstrict-prototypes -Werror \
       -Wpacked -std=c90 -ansi -pedantic -O3 -Iinclude
LDLIBS=-lm

SRCS=src/nn/nn.c src/neat/population.c src/neat/species.c src/neat/genome.c
OBJS=$(SRCS:.c=.o)

all: build

$(LIB): $(OBJS)
	$(AR) $@ $^
	$(RANLIB) $@

.PHONY: build
build: $(LIB)
	"$(MAKE)" -C test LIB="../$(LIB)"

.PHONY: example
example: $(LIB)
	"$(MAKE)" -C example LIB="../$(LIB)"

.PHONY: test
test: $(LIB)
	"$(MAKE)" -C test test LIB="../$(LIB)"

.PHONY: clean
clean:
	$(RM) $(OBJS) $(LIB)
	"$(MAKE)" -C example clean
	"$(MAKE)" -C test clean
