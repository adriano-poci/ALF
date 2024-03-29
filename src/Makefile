#======================
# Fortran 90 compiler
# (Uncomment only one)
#======================
# GNU
#F90 = gfortran
#---------------------
# Intel
#F90 = ifort 
#---------------------
#MPI
F90 = mpifort 
#F90 = mpif90
#========================
# Compiler Optimizations
#  (Uncomment only one)
#========================
# GNU
F90FLAGS = -O3 -fno-strict-overflow -cpp #-mcmodel=large
#F90FLAGS = -O3 -fno-strict-overflow  #for odyssey
#---------------------
# Intel 
#F90FLAGS = -O3 -assume byterecl
# Intel (EM64T/AMD64 processor)
#F90FLAGS = -O3 -funroll-loops -xP
#---------------------

PROGS = alf write_a_model spec_from_sum ml_from_mcmc

COMMON =  nrtype.o alf_vars.o nr.o nrutil.o ran_state.o alf_utils.o getmodel.o \
	setup.o linterp.o velbroad.o tsum.o add_response.o \
	contnormspec.o set_pinit_priors.o str2arr.o locate.o gasdev.o \
	ran1.o lfit.o gaussj.o covsrt.o powell.o linmin.o sort.o \
	mnbrak.o brent.o getm2l.o read_data.o func.o init_random_seed.o \
	getmass.o maskemlines.o getvelz.o emcee_advance_mpi.o update_runtot.o \
	linterp3.o free_workers.o function_parallel_map.o myran.o getindx.o \
	vacairconv.o

all : $(PROGS)

clean :
	rm -rf *.o *.mod *.MOD *~

alf : alf.o $(COMMON) 
	$(F90) $(F90FLAGS) -o ../bin/alf.exe alf.o $(COMMON)

write_a_model : write_a_model.o $(COMMON) 
	$(F90) -o ../bin/write_a_model.exe write_a_model.o $(COMMON)

spec_from_sum : spec_from_sum.o $(COMMON) 
	$(F90) -o ../bin/spec_from_sum.exe spec_from_sum.o $(COMMON)

ml_from_mcmc : ml_from_mcmc.o $(COMMON) 
	$(F90) -o ../bin/ml_from_mcmc.exe ml_from_mcmc.o $(COMMON)

alf.o : alf_vars.o nrtype.o nr.o alf_utils.o ran_state.o nrutil.o

write_a_model.o : alf_vars.o nrtype.o nr.o alf_utils.o ran_state.o nrutil.o

spec_from_sum.o : alf_vars.o nrtype.o nr.o alf_utils.o ran_state.o nrutil.o

ml_from_mcmc.o : alf_vars.o nrtype.o nr.o alf_utils.o ran_state.o nrutil.o

ran_state.o: nrutil.o

%.o : nr/%.f90
	$(F90) $(F90FLAGS) -o $@ -c $<

%.o : %.f90
	$(F90) $(F90FLAGS) -o $@ -c $<


