NAME=neat-test

CFLAGS=-g -Wall -Wextra -Wmissing-prototypes -Wstrict-prototypes -Werror \
       -std=c90 -ansi -pedantic -O3 -I../include
LDLIBS=-lm -L. ${LIB}

SRCS=test.c
OBJS=$(SRCS:.c=.o)

all: $(NAME)

$(NAME): $(OBJS) ${LIB}
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test
test: $(NAME)
	./$(NAME)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
