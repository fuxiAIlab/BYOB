import os
import argparse

import ray
from ray import tune
from byob.models.sim_env import BundleEnv
# from byob.models.sim_env_v1 import BundleEnv
from byob.models.sim_model import TFBundleModel, TorchBundleModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

from byob.config import DATA_CONFIG, MODEL_CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-reward", type=float, default=100.0)
parser.add_argument("--stop-timesteps", type=int, default=1000000)
parser.add_argument('--dataset', type=str, default='movielens', choices=('movielens', 'yoochoose'))
parser.add_argument('--pool_size', type=int, default=20, help='pool size (default: 20)')
parser.add_argument('--bundle_size', type=int, default=3, help='bundle size (default: 3)')
parser.add_argument('--env_mode', type=str, default='train', choices=('train', 'test'))
# parser.add_argument('--rew_mode', type=str, default='metric', choices=('item', 'compat', 'bundle', 'metric'))
parser.add_argument('--rew_mode', type=str, default='metric')
parser.add_argument('--metric', type=str, default='recall', choices=('precision', 'precision_plus', 'recall'))
parser.add_argument("--encoder", action="store_true")
parser.add_argument("--concat", action="store_true")
parser.add_argument("--embed-pretrain", action="store_true")
parser.add_argument("--fine-tune", action="store_true")

if __name__ == "__main__":

    args = parser.parse_args()
    args.run = 'PPO'
    args.torch = True
    args.stop_timesteps = int(1e6)
    # args.pool_size = 100
    # args.bundle_size = 5
    # args.rew_mode = 'item'
    print(vars(args))

    ray.init()

    env_config = {
        'dataset': args.dataset,
        'pool_size': args.pool_size,
        'bundle_size': args.bundle_size,
        'env_mode': args.env_mode,
        'rew_mode': args.rew_mode,
        'metric': args.metric,
        'item_model': 'NCF',
        'bundle_model': 'NCF',
        'compat_model': 'SG',
        'num_features': 1,
        'device': 'cpu'
    }
    env_config.update(DATA_CONFIG[env_config['dataset']])
    env_config['vocab_size'] = env_config['n_item']
    print(env_config)

    custom_model_config = {
        'env_config': env_config,
        'model': 'BYOB',
        'n_user': env_config['n_user'],
        'n_item': env_config['n_item'],
        'pool_size': env_config['pool_size'],
        'bundle_size': env_config['bundle_size'],
        'encoder': args.encoder,  # pool transformer encoder
        'concat': args.concat,  # concat bundle features
        'embed_path': None,  # use pretrained embedding
        'fine_tune': args.fine_tune,  # fine-tuning embedding
    }
    custom_model_config.update(MODEL_CONFIG[custom_model_config['model']])
    # custom_model_config['encoder'] = True
    # custom_model_config['concat'] = True
    # https://www.thinbug.com/q/54924582
    # https://blog.csdn.net/wen_fei/article/details/83117324
    if args.embed_pretrain:
        custom_model_config['embed_path'] = '/root/reclib/byob/output/models/movielens-SkipGramModel.npy'
    print(custom_model_config)

    register_env("sim_env", lambda _: BundleEnv(env_config))

    ModelCatalog.register_custom_model("sim_model", TorchBundleModel if args.torch else TFBundleModel)

    if args.run == "DQN":
        cfg = {
            # TODO(ekl) we need to set these to prevent the masked values
            # from being further processed in DistributionalQModel, which
            # would mess up the masking. It is possible to support these if we
            # defined a custom DistributionalQModel that is aware of masking.
            "hiddens": [],
            "dueling": False,
            "exploration_config": {
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.

                # For soft_q, use:
                # "exploration_config" = {
                #   "type": "SoftQ"
                #   "temperature": [float, e.g. 1.0]
                # }
            },
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": int(1e6)
        }
    else:
        cfg = {}

    config = dict(
        {
            "env": "sim_env",
            "model": {
                'fcnet_hiddens': [1024, 512, 256],
                'fcnet_activation': 'tanh',
                "custom_model": "sim_model",
                "custom_model_config": custom_model_config
            },
            "num_gpus": 1,
            "num_workers": 12,
            "framework": "torch" if args.torch else "tf",
        },
        **cfg)

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # https://docs.ray.io/en/master/rllib-models.html#variable-length-parametric-action-spaces
    # https://github.com/ray-project/ray/blob/master/rllib/examples/parametric_actions_cartpole.py
    # results = tune.run(args.run, stop=stop, config=config, verbose=1)
    results = tune.run(args.run, stop=stop, config=config, checkpoint_freq=10, checkpoint_at_end=True, verbose=1)

    # https://github.com/ray-project/ray/issues/4569
    # https://github.com/ray-project/ray/issues/8827
    # ckpt_path = 'PPO_sim_env_ef654_00000_0_2021-04-07_23-35-52'  # item
    # # ckpt_path = 'PPO_sim_env_a9d08_00000_0_2021-04-08_00-38-20'  # compat
    # # ckpt_path = 'PPO_sim_env_7ab6b_00000_0_2021-04-08_01-41-27'  # bundle
    # ckpt_path = '~/ray_results/PPO/%s/checkpoint_100/checkpoint-100' % ckpt_path
    # results = tune.run(args.run, stop=stop, config=config, checkpoint_freq=10, checkpoint_at_end=True, verbose=1, restore=ckpt_path)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
