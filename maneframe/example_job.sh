#!/usr/bin/env bash
#SBATCH -J job_name
#SBATCH -o result/std_out_file_name%j.txt
#SBATCH -e result/std_err_file_name-err%j.txt
#SBATCH -p gpgpu-1 --gres=gpu:8 --mem=60G
#SBATCH -t 10080
#SBATCH --mail-user user_name@smu.edu
#SBATCH --mail-type=ALL
#SBATCH -s

python script_name.py
