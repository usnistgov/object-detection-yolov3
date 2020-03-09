#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --job-name=y3
#SBATCH --time=72:0:0
#SBATCH --exclusive
#SBATCH -o y3-%N.%j.out
#SBATCH --mail-user=michael.majurski@nist.gov
#SBATCH --mail-type=ALL


source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf2

# job configuration
test_every_n_steps=10000
batch_size=8 # 8x across the gpus
learning_rate=1e-4

train_lmdb_file="train-yolov3.lmdb"
test_lmdb_file="test-yolov3.lmdb"

input_data_directory="/wrk/mmajursk/"
output_directory="/wrk/mmajursk/yolov3"

experiment_name="y3-$(date +%Y%m%dT%H%M%S)"

# MODIFY THESE OPTIONS
# **************************

echo "Experiment: $experiment_name"
scratch_dir="/scratch/${SLURM_JOB_ID}"

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
term_handler()
{
        echo "function term_handler called.  Cleaning up and Exiting"
        rm -rf ${scratch_dir}/*
        # Do nothing
        exit -1
}

results_dir="$output_directory/$experiment_name"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

mkdir -p "$results_dir/src"
cp -r . "$results_dir/src"


# associate the function "term_handler" with the TERM signal
trap 'term_handler' TERM
# copy data to node
echo "Copying data to Node"
cp -r ${input_data_directory}/${train_lmdb_file} ${scratch_dir}/${train_lmdb_file}
cp -r ${input_data_directory}/${test_lmdb_file} ${scratch_dir}/${test_lmdb_file}
echo "data copy to node complete"
echo "Working directory contains: "
ls ${scratch_dir}/



# launch training script with required options
echo "Launching Training Script"
python train.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$scratch_dir/$train_lmdb_file" --test_database="$scratch_dir/$test_lmdb_file" --output_dir="$results_dir" --learning_rate=${learning_rate}


echo "cleaning up scratch space"
rm -rf ${scratch_dir}/*
echo "Job completed"








