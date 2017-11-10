all: gtk-visualization flappy-bird

.PHONY: gtk-visualization
gtk-visualization: $(LIB)
	"$(MAKE)" -C gtk-visualization LIB="../$(LIB)"

.PHONY: flappy-bird
flappy-bird: $(LIB)
	"$(MAKE)" -C flappy-bird LIB="../$(LIB)"
