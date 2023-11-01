#!/encs/bin/tcsh

# Give job a name
#$ -N CC_2023_pre_CFR

# Set output directory to current
#$ -cwd

# Send an email when the job starts, finishes or if it is aborted.
#$ -m bea

# Request GPU
# #$ -l gpu=2

#$ -pe smp 32

# Request CPU with maximum memory size = 500GB
#$ -l h_vmem=500G

#sleep 30

# Specify the output file name in our case, its comment, and the system will generate the file with the same name as the job
# -o name.qlog


conda activate  /speed-scratch/m_daragh/venv/py31013_pc310

# example 
python buckets_plots.py

conda deactivate
