


TESTFILES = count-test

OBJFILES = count

BINFILES =

CXXFLAGS += -Wall

LDFLAGS = -g
LDLIBS =
CXX = g++
CC = $(CXX)
RANLIB = ranlib
AR = ar

depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk

.depend.mk: depend
-include .depend.mk

