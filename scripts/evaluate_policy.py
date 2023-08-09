'''Template training/plotting/testing script.'''

import os
import pickle
from functools import partial

import wandb
import torch
import numpy as np

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video
from safe_control_gym.controllers.ppo import cbf_ppo

def record_obs_hist(model: cbf_ppo.CBFPPO,
        env=None,
        render=False,
        n_episodes=10,
        verbose=False,
        ):
    '''Runs evaluation with current policy.'''
    model.agent.eval()
    model.obs_normalizer.set_read_only()
    env = model.env

    obs, info = env.reset()
    ep_obs_hist = [obs]
    obs = model.obs_normalizer(obs)
    full_obs_hist = []

    n_episodes_complete = 0

    while n_episodes_complete < n_episodes:
        action = model.select_action(obs=obs, info=info)
        obs, _, done, info = env.step(action)
        ep_obs_hist.append(obs)

        if verbose:
            print(f'obs {obs} | act {action}')
        if done:
            n_episodes_complete += 1
            n_steps = len(ep_obs_hist)
            ep_obs_hist += [np.zeros_like(ep_obs_hist[0], dtype=np.float32)] * (301 - n_steps)
            full_obs_hist.append(np.array(ep_obs_hist))
            obs, _ = env.reset()
            ep_obs_hist = [obs]

        obs = model.obs_normalizer(obs)

    return np.array(full_obs_hist)

def test_policy(config):
    '''Run the (trained) policy/controller for evaluation.

    Usage
        * use with `--func test`.
        * to test policy from a trained model checkpoint, additionally use
            `--restore {dir_path}` where `dir_path` is folder to the trained model.
        * to test un-trained policy (e.g. non-learning based), use as it is.
    '''
    # Evaluation setup.
    set_device_from_config(config)
    if config.set_test_seed:
        # seed the evaluation (both controller and env) if given
        set_seed_from_config(config)
        env_seed = config.seed
    else:
        env_seed = None
    # Define function to create task/env.
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=False,
                         checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                         output_dir=config.output_dir,
                         use_gpu=config.use_gpu,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()
    if config.restore:
        control_agent.load(os.path.join(config.restore, 'model_latest.pt'))
    # Test controller.
    print("Evaluating on {} episodes".format(config.algo_config.eval_batch_size))

    full_obs_hist = record_obs_hist(control_agent, n_episodes=config.algo_config.eval_batch_size, verbose=config.verbose)
    # Save evaluation results.

    np.save(os.path.join(config.restore, f'obs_hist.npy'), full_obs_hist)
    control_agent.close()
    print('Evaluation done.')

if __name__ == '__main__':
    # Make config.
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--thread', type=int, default=0, help='number of threads to use (set by torch).')
    fac.add_argument('--render', action='store_true', help='if to render in policy test.')
    fac.add_argument('--verbose', action='store_true', help='if to print states & actions in policy test.')
    fac.add_argument('--use_adv', action='store_true', help='if to evaluate against adversary.')
    fac.add_argument('--set_test_seed', action='store_true', help='if to set seed when testing policy.')
    fac.add_argument('--eval_output_dir', type=str, default='', help='folder path to save evaluation results.')
    fac.add_argument('--eval_output_path', type=str, default='test_results.pkl', help='file path to save evaluation results.')
    config = fac.merge()
    # System settings.
    if config.thread > 0:
        # E.g. set single thread for less context switching
        torch.set_num_threads(config.thread)
    # Execute.

    test_policy(config)