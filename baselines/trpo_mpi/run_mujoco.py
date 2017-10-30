#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import os.path as osp
import sys
import argparse
import logging

import gym
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi

def train(env_id, num_timesteps, seed):

    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def sample(env_id, seed, load_model_path):

    sess = U.single_threaded_session()
    sess.__enter__()

    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir()) 
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.sample(env, policy_fn, timesteps_per_batch=1024, load_model_path=load_model_path)
    env.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--task', type=str, choices=['train', 'sample'], default='train')
    parser.add_argument('--load_model_path', type=str)
    args = parser.parse_args()
    logger.configure()
    if args.task == 'train':
        train(args.env, args.num_timesteps, args.seed)
    elif args.task == 'sample':
        sample(args.env, args.seed, args.load_model_path)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
