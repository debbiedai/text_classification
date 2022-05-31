#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --job-name=run_tc
#SBATCH --account=nal_genomics
#SBATCH --mail-user=<xxxxxxxx@gmail.com>
#SBATCH --mail-type=NONE
#SBATCH --time=22:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=48


python main.py