NAME=neat-flappy

CFLAGS=-g -Wall -Wextra -Wmissing-prototypes -Wstrict-prototypes -Werror \
       -std=gnu90 -pedantic -O3 -I../../include
LDLIBS=-lm -lncurses

SRCS=flappy.c
OBJS=$(SRCS:.c=.o)

all: $(NAME)

$(NAME): $(OBJS) ${LIB}
	$(CC) $(LDFLAGS) -o $@ -L. $^ $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
