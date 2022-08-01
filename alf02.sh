#!/bin/bash
#SBATCH -A oz059
#SBATCH --job-name="alfNGC4365"
#SBATCH --time=2-00:00
#SBATCH -D "/fred/oz059/poci/alf"
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@students.mq.edu.au
#SBATCH --array=0-461
#SBATCH -o /dev/null # Standard out goes to this file
#SBATCH -e /dev/null # Standard err goes to this file

module load gcc/9.2.0
module load openmpi/4.0.5
module load anaconda3/2021.05
module load python/3.10.4

export ALF_HOME="/fred/oz059/poci/alf/"

cd ${ALF_HOME}
declare idx=$(printf %04d $((${SLURM_ARRAY_TASK_ID} + 2000)))
mpirun --oversubscribe -np ${SLURM_CPUS_PER_TASK} ./bin/alf.exe "ngc4365_${idx}" >& "out_ngc4365_${idx}.log"
