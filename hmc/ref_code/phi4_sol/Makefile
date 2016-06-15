IDIR =.
CC=gcc
CFLAGS=-O3  -std=c89 -Wall -Werror -pedantic -fstrict-aliasing -I$(IDIR)

ODIR=.

LIBS=-lm

_DEPS = phi4.h ranlxd.h lattice.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = hopping.o  ranlxd.o phi4.o measure.o gauss_rand.o hmc.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS)
		$(CC) -c -o $@ $< $(CFLAGS)

phi4: $(OBJ)
		gcc -o $@ $^ $(CFLAGS) $(LIBS)


.PHONY: clean

clean:
		rm -f $(ODIR)/*.o  core 


