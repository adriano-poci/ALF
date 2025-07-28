#!/bin/bash -l
#SBATCH -A durham
#SBATCH -p cosma
#SBATCH --job-name="alf_SNL1_NFM_1arcs"
#SBATCH --time=0-48:00
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL
#SBATCH --mail-user=adriano.poci@durham.ac.uk
#SBATCH -o /dev/null # Standard out goes to piped file
#SBATCH -e SNL1_NFM_error.log # Standard err to galaxy
#SBATCH --open-mode=append

module load gnu_comp
module load python/3.10.7
module load openmpi/20190429
module load cmake/3.18.1
source ${HOME}/.bashrc

export ALF_HOME=/cosma5/data/durham/dc-poci1/alf/

# Compile clean version of `alf`
cd ${ALF_HOME}src
cp alf.f90.perm alf.f90
# Remove prior placeholders on velz
sed -i "/prlo%velz = -999./d" alf.f90
sed -i "/prhi%velz = 999./d" alf.f90
make all && make clean
cd ${ALF_HOME}
# Run aperture fit
mpirun --oversubscribe -np ${SLURM_CPUS_PER_TASK} ./bin/alf.exe "SNL1_NFM_1arcs" 2>&1 | tee -a "SNL1_NFM/out_1arcs.log"
