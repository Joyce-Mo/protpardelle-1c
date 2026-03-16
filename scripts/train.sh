#!/bin/bash           # the shell language when run outside of the job scheduler
#                     # lines starting with #$ is an instruction to the job scheduler
#$ -S /bin/bash       # the shell language when run via the job scheduler [IMPORTANT]
#$ -cwd               # job should run in the current working directory
#$ -j y               # STDERR and STDOUT should be joined
#$ -l mem_free=1G     # job requires up to 1 GiB of RAM per slot
#$ -l scratch=2G      # job requires up to 2 GiB of local /scratch space
#$ -l h_rt=48:00:00   # job requires up to 48 hours of runtime
##$ -t 1-10           # array job with 10 tasks (remove first '#' to enable)
#$ -r y               # if job crashes, it should be restarted
#$ -l gpu=4          # job requires 4 GPUs 

## If you array jobs (option -t), this script will run T times, once per task.
## For each run, $SGE_TASK_ID is set to the corresponding task index (here 1-10).
## To configure different parameters for each task index, one can use a Bash 
## array to map from the task index to a parameter string.

## All possible parameters
# params=(1bac 2xyz 3ijk 4abc 5def 6ghi 7jkl 8mno 9pqr 10stu)

## Select the parameter for the current task index
## Arrays are indexed from 0, so we subtract one from the task index
# param="${params[$((SGE_TASK_ID - 1))]}"

date 
hostname

if command -v module >/dev/null 2>&1; then
    module load gcc/12.4.0
fi

if [ -z "$ENV_DIR" ]; then
    conda activate protpardelle
else
    source "$ENV_DIR"/protpardelle/bin/activate
fi

PROJECT_ROOT_DIR=$(python -c "from protpardelle.env import PROJECT_ROOT_DIR; print(PROJECT_ROOT_DIR)")

model_name=$89
output_dir=$2

if [[ "${2:-}" == "--debug" ]]; then
    debug="--debug"
fi

config_path="${PROJECT_ROOT_DIR}/examples/training/cc${model_name}.yaml"
project_name="protpardelle-1c"
wandb_id="your-wandb-id"

WANDB__SERVICE_WAIT=300 python -m protpardelle.train \
    --config-path "${config_path}" \
    --output-dir "${output_dir}" \
    --project-name "${project_name}" \
    --wandb-id "${wandb_id}" \
    --exp-name "${model_name}" \
    ${debug:+$debug}

## End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"  # This is useful for debugging and usage purposes,
                                          # e.g. "did my job exceed its memory request?"
