all: build

.PHONY: build
build:
	$(MAKE) -C example
	$(MAKE) -C test

.PHONY: test
test: build
	$(MAKE) -C test test

clean:
	$(MAKE) -C example clean
	$(MAKE) -C test clean
