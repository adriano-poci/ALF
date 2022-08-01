#!/cosma/local/Python/3.9.1-cosma7/bin/python3
#SBATCH -A durham
#SBATCH -p cosma
#SBATCH --job-name="slurm_alf_Spec"
#SBATCH --time=0-12:00
#SBATCH -D "/cosma5/data/durham/dc-poci1/alf"
#SBATCH --output="/cosma5/data/durham/dc-poci1/alf/slurm_alfSpec.log"
#SBATCH --error="/cosma5/data/durham/dc-poci1/alf/slurm_alfSpec.log"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@durham.ac.uk
"""
    slurm_alfSpec.py
    Adriano Poci
    Durham University

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module executes some function in the `SLURM` queueing environment

    Authors
    -------
    Adriano Poci <adriano.poci@durham.ac.uk>

History
-------
v1.0:	14 July 2022
"""

# Custom modules
import alf_MUSE as am

am.makeSpecFromSum('NGC4365', SN=100, NMP=16)
am.afh('NGC4365', SN=100, full=True, NMP=16, band='F814W', photFilt='WFPC2.F814W')