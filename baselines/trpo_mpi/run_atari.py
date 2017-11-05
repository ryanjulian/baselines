#!/usr/bin/env python3
import argparse
import logging
import os

import gym
from mpi4py import MPI

import baselines.common.tf_util as U
from baselines import logger
from baselines import bench
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.trpo_mpi import trpo_mpi
from baselines.trpo_mpi.no_sharing_cnn_policy import NoSharingCnnPolicy


def train(env_id, num_timesteps, seed, rank):
    with U.single_threaded_session() as sess:
        workerseed = seed + 10000 * rank
        set_global_seeds(workerseed)

        env = make_atari(env_id)
        env = bench.Monitor(env,
                            logger.get_dir()
                            and os.path.join(logger.get_dir(), str(rank)))
        env.seed(workerseed)
        env = wrap_deepmind(env)
        env.seed(workerseed)

        policy_fn = lambda name, ob_space, ac_space: NoSharingCnnPolicy(name=name,
                ob_space=ob_space, ac_space=ac_space)
        trpo_mpi.learn(
            env,
            policy_fn,
            timesteps_per_batch=512,
            max_kl=0.001,
            cg_iters=10,
            cg_damping=1e-3,
            max_timesteps=int(num_timesteps * 1.1),
            gamma=0.98,
            lam=1.0,
            vf_iters=3,
            vf_stepsize=1e-4,
            entcoeff=0.00)

        env.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    gym.logger.setLevel(logging.WARN)

    train(
        args.env, num_timesteps=args.num_timesteps, seed=args.seed, rank=rank)


if __name__ == "__main__":
    main()
