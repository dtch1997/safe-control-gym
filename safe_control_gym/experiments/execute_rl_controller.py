'''Template training/plotting/testing script.'''

import os
import pickle
from functools import partial

import wandb
import yaml
import torch

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video


def train(config):
    '''Training template.

    Usage:
        * to start training, use with `--func train`.
        * to restore from a previous training, additionally use `--restore {dir_path}`
            where `dir_path` is the output folder from previous training.
    '''
    # Experiment setup.
    print("Setting up experiment")
    print(yaml.safe_dump(config))
    config_savepath = os.path.join(config.output_dir, 'config.yaml')
    with open(config_savepath, 'w') as f:
        yaml.dump(config, f)
    if not config.restore:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    name = f"{config.algo}_{config.task}_{config.task_config.task}_{config.algo_config.safety_coef}"
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        name=name,
        config=config,
        sync_tensorboard=True
    )

    # Define function to create task/env.
    print("Creating env")
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    print("Creating controller")
    control_agent = make(config.algo,
                         env_func,
                         training=True,
                         checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                         output_dir=config.output_dir,
                         use_gpu=config.use_gpu,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()
    if config.restore:
        control_agent.load(os.path.join(config.restore, 'model_latest.pt'))
    # Training.
    control_agent.learn()
    control_agent.close()
    # Save models 
    wandb.save(os.path.join(config.output_dir, 'model_latest.pt'), base_path = config.output_dir)
    wandb.save(os.path.join(config.output_dir, 'model_best.pt'), base_path = config.output_dir)
    wandb.save(config_savepath, base_path = config.output_dir)
    print('Training done.')


def make_plots(config):
    '''Produces plots for logged stats during training.

    Usage
        * use with `--func plot` and `--restore {dir_path}` where `dir_path` is
            the experiment folder containing the logs.
        * save figures under `dir_path/plots/`.
    '''
    # Define source and target log locations.
    log_dir = os.path.join(config.output_dir, 'logs')
    plot_dir = os.path.join(config.output_dir, 'plots')
    mkdirs(plot_dir)
    plot_from_logs(log_dir, plot_dir, window=3)
    print('Plotting done.')


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
    results = control_agent.run(n_episodes=config.algo_config.eval_batch_size,
                                render=config.render,
                                verbose=config.verbose)
    # Save evalution results.
    if config.eval_output_dir is not None and config.eval_output_dir:
        eval_output_dir = config.eval_output_dir
    else:
        eval_output_dir = os.path.join(config.output_dir, 'eval')
    os.makedirs(eval_output_dir, exist_ok=True)
    # test trajs and statistics
    eval_path = os.path.join(eval_output_dir, config.eval_output_path)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, 'wb') as f:
        pickle.dump(results, f)
    ep_lengths = results['ep_lengths']
    ep_returns = results['ep_returns']
    ep_constraint_violations = results['ep_constraint_violations']
    msg = 'eval_ep_length {:.2f} +/- {:.2f}\n'.format(ep_lengths.mean(), ep_lengths.std())
    msg += 'eval_ep_return {:.3f} +/- {:.3f}\n'.format(ep_returns.mean(), ep_returns.std())
    msg += 'eval_ep_constraint_violation {:.3f} +/- {:.3f}\n'.format(ep_constraint_violations.mean(), ep_constraint_violations.std())
    print(msg)
    if 'frames' in results:
        save_video(os.path.join(eval_output_dir, 'video.gif'), results['frames'])
    if 'obs_hist' in results:
        import numpy as np
        np.save(os.path.join(config.restore, 'obs_hist.npy'), results['obs_hist'])
    control_agent.close()
    print('Evaluation done.')


MAIN_FUNCS = {'train': train, 'plot': make_plots, 'test': test_policy}


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
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception(f'Main function {config.func} not supported.')
    func(config)
