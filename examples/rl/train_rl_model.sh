#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_3D'

# TASK='stab'
TASK='track'

ALGO='cbf_ppo'
# ALGO='sac'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
for seed in 0 1 2 3 4
do 
    for safety_coef in 0 1 10 
    do
        python3 ../../safe_control_gym/experiments/execute_rl_controller.py \
            --algo ${ALGO} \
            --task ${SYS_NAME} \
            --overrides \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
            --output_dir ./ \
            --tag ${ALGO}_${SYS}_${TASK}/ \
            --seed $seed \
            --kv_overrides algo_config.safety_coef=$safety_coef wandb.group='expt'
    done
done