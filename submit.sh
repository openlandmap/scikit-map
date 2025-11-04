#!/bin/bash
#SBATCH --partition=cpu               							 # Partition name (cpu, gpu, nas, all)
#SBATCH --ntasks-per-node=1          								 # Do not change it
#SBATCH --ntasks=1                  								 # Do not change it
#SBATCH --job-name=wri_gpp		 		 	 			 					 # Job name
#SBATCH --array=1-3                								 # Number of server to use
#SBATCH --mail-type=ALL            									 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=void   # Where to send mail
#SBATCH --output=%x_%N.log  												 # Do not change it
#SBATCH --exclude=	 # Servers to exclude (landmark,primus)

# Slurm helper functions
source /mnt/slurm/jobs/ogh_slurm.sh

# Total number of ids to split and process across the servers
#N_IDS=817 # BR
N_IDS=3

#N_IDS=$(cat /mnt/slurm/jobs/wri_gpp/bad_guys.txt | wc -l) 
# Python docker container
DOCKER_IMAGE=192.168.49.30:5000/pygeo-ide:v3.8.16-mkl-gdal362-pasture_class
LANG=Python

# R docker container
#DOCKER_IMAGE=opengeohub/rgeo:v4.1.1-mkl-gdal314
#LANG=R

execute
