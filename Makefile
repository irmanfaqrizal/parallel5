CC = mpicc
CFLAGS = 
LDFLAGS =

EXEC = 	matrix_multiplication_seq.run

HEADER_FILES = $(wildcard *.h)

EXTRA_SRC = utils.c

RAND_INIT=1

CHECK_CORRECT=0

EVAL_PERF=1


ifeq ($(EVAL_PERF), 0)
ifeq ($(CHECK_CORRECT), 0)
$(error Wrong combination of options. EVAL_PERF or CHECK_CORRECT need to be set)
endif
endif

ifeq ($(RAND_INIT), 1)
$(info Initialization of the matrices is random)
CONFIG_FLAGS += -DRINIT
endif

ifeq ($(CHECK_CORRECT), 1)
$(info Computation if checked for correctness)
CONFIG_FLAGS += -DCHECK_CORRECTNESS
endif

ifeq ($(EVAL_PERF), 1)
$(info Performance is evaluated)
CONFIG_FLAGS += -DPERF_EVAL
endif


all: $(EXEC)

%.run: $(HEADER_FILES)

%.run: %.c $(EXTRA_SRC)
	$(CC) -o $@ $^ $(CONFIG_FLAGS)

clean:
	rm -f $(EXEC) *.o *~

.PHONY: clean
