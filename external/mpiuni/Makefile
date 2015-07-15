all: lib

lib: mpi.c
	g++ -I. -c mpi.c -DPETSC_HAVE_FORTRAN_UNDERSCORE
	ar cr libmpiuni.a mpi.o
	${RM} mpi.o

clean:
	${RM} libmpiuni.a
