#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import mujoco_py  # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850

import argparse
import logging
import os

import gym
from mpi4py import MPI

import baselines.common.tf_util as U
from baselines import logger
from baselines import bench
from baselines.common import set_global_seeds
from baselines.common.mpi_fork import mpi_fork
from baselines.trpo_mpi import trpo_mpi
from baselines.trpo_mpi.mlp_policy import MlpPolicy


def train(env_id, num_timesteps, seed, rank):
    with U.single_threaded_session() as sess:
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)

        env = gym.make(env_id)
        env = bench.Monitor(env,
                            logger.get_dir()
                            and os.path.join(logger.get_dir(), str(rank)))
        env.seed(workerseed)

        policy_fn = lambda name, ob_space, ac_space: MlpPolicy(name=name,
                ob_space=ob_space, ac_space=ac_space,
                hid_size=32, num_hid_layers=2)
        trpo_mpi.learn(
            env,
            policy_fn,
            timesteps_per_batch=1024,
            max_kl=0.01,
            cg_iters=10,
            cg_damping=0.1,
            max_timesteps=num_timesteps,
            gamma=0.99,
            lam=0.98,
            vf_iters=5,
            vf_stepsize=1e-3)

        env.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.set_level(logger.DISABLED)
    gym.logger.setLevel(logging.WARN)

    train(
        args.env, num_timesteps=args.num_timesteps, seed=args.seed, rank=rank)


if __name__ == '__main__':
    main()
