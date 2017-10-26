NAME=neat-drawing

RM=rm -rf
CFLAGS=-g -Wall -pedantic -O1 -I../include \
       `pkg-config --cflags gtk+-3.0`
LDLIBS=-fopenmp -lm \
       `pkg-config --libs gtk+-3.0`

SRCS=drawing.c ../src/nn/nn.c \
     ../src/neat/population.c ../src/neat/species.c ../src/neat/genome.c
OBJS=$(SRCS:.c=.o)

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(NAME)
