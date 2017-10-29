NAME=neat-drawing

CFLAGS=-g -Wall -Wextra -Wmissing-prototypes -Wstrict-prototypes -Werror \
       -std=c99 -pedantic -O3 -I../include \
       `pkg-config --cflags gtk+-3.0`
LDLIBS=-lm -L. \
       `pkg-config --libs gtk+-3.0`

SRCS=drawing.c
OBJS=$(SRCS:.c=.o)

all: $(NAME)

$(NAME): $(OBJS) ${LIB}
	$(CC) $(LDFLAGS) -o $@ $(LDLIBS) $^

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
