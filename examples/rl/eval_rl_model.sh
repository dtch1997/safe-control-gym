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
python3 ../../safe_control_gym/experiments/execute_rl_controller.py \
    --func test \
    --restore models/cbf_ppo/safety_coef_0

# Train the unsafe controller/agent.
python3 ../../safe_control_gym/experiments/execute_rl_controller.py \
    --func test \
    --restore models/cbf_ppo/safety_coef_10