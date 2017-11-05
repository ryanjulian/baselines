#!/usr/bin/env python3
import argparse
import logging

import gym

from baselines import bench
from baselines import logger
from baselines.common import tf_util as U
from baselines.common import set_global_seeds
from baselines.ppo1 import pposgd_simple
from baselines.ppo1.mlp_policy import MlpPolicy


def train(env_id, num_timesteps, seed):
    with U.make_session(num_cpu=1) as sess:
        set_global_seeds(seed)

        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        env.seed(seed)

        policy_fn = lambda name, ob_space, ac_space: MlpPolicy(name=name,
            ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
        pposgd_simple.learn(
            env,
            policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2,
            entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
        )

        env.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()

    logger.configure()
    gym.logger.setLevel(logging.WARN)

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
