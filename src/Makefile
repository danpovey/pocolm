
all:


TESTFILES = count-test

OBJFILES = count.o lm-state.o lm-state-derivs.o

BINFILES = get-text-counts get-int-counts print-int-counts \
	merge-counts print-counts discount-counts print-float-counts \
    discount-counts-1gram merge-float-counts compute-probs \
    print-float-derivs perturb-float-counts discount-counts-1gram-backward \
    print-derivs perturb-counts discount-counts-backward \
    merge-counts-backward split-int-counts sum-count-derivs sum-float-derivs \
    split-int-counts-by-order float-counts-to-pre-arpa pre-arpa-to-arpa \
    float-counts-to-float-stats float-counts-estimate float-counts-to-histories \
    histories-to-null-counts print-null-counts float-counts-prune

$(BINFILES): $(OBJFILES)


all: $(TESTFILES) $(BINFILES)

clean:
	-rm *.o $(BINFILES) $(TESTFILES)

CXXFLAGS += -Wall -g
LDFLAGS += -g

ifeq ($(shell uname), Darwin)
 CXXFLAGS += -stdlib=libstdc++
 LDFLAGS += -stdlib=libstdc++
endif

LDLIBS =
CXX = g++
CC = $(CXX)
RANLIB = ranlib
AR = ar

depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk

.depend.mk: depend
-include .depend.mk

