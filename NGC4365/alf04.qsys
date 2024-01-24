#!/bin/bash -l
#SBATCH -A durham
#SBATCH -p cosma
#SBATCH --job-name="alf_NGC4365_SN100"
#SBATCH -D "/cosma5/data/durham/dc-poci1/alf/NGC4365"
#SBATCH --time=0-48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3000
#SBATCH --array=0-330
#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL
#SBATCH --mail-user=adriano.poci@durham.ac.uk
#SBATCH -o "/cosma5/data/durham/dc-poci1/alf/NGC4365/out.log" # Standard out to galaxy
#SBATCH -e "/cosma5/data/durham/dc-poci1/alf/NGC4365/out.log" # Standard err to galaxy
#SBATCH --open-mode=append

source ${HOME}/.bashrc

module load gnu_comp
module load python/3.10.7
module load openmpi/20190429
module load cmake/3.18.1
export ALF_HOME=/cosma5/data/durham/dc-poci1/alf/

cd ${ALF_HOME}
declare idx=$(printf %04d $((${SLURM_ARRAY_TASK_ID} + 1320)))
mpirun --oversubscribe -np ${SLURM_CPUS_PER_TASK} ./NGC4365/bin/alf.exe "NGC4365_SN100_${idx}" 2>&1 | tee -a "NGC4365/out_${idx}.log"
