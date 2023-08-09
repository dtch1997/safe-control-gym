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
    for safety_coef in 10  
    do
        for bounded in True False
        do 
            python3 ../../safe_control_gym/experiments/execute_rl_controller.py \
                --algo ${ALGO} \
                --task ${SYS_NAME} \
                --overrides \
                    ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                    ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                --output_dir ./ \
                --tag bounded=${bounded}_seed=${seed}/ \
                --seed $seed \
                --kv_overrides \
                    wandb.group=${ALGO}_${SYS}_${TASK}_ablations \
                    algo_config.safety_coef=$safety_coef \
                    algo_config.bounded=$bounded
        done
    done
done