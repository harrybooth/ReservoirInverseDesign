#!/bin/bash
#SBATCH --job-name=T2S_ParamSweep_MT
#SBATCH --ntasks=250
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --partition=ncpu
#SBATCH --mail-user=harry.booth@crick.ac.uk
#SBATCH --mail-type=ALL

# Initialise Julia
module load Julia/1.11.4-linux-x86_64

export JULIA_DEPOT_PATH=/camp/home/boothh/home/users/boothh/julia_depot
export TMPDIR=/camp/home/boothh/home/users/boothh/tmp

julia GenerateMolecularProfiles.jl