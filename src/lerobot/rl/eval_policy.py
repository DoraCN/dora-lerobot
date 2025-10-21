# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so101_leader,  # noqa: F401
)

from .gym_manipulator import make_robot_env

logging.basicConfig(level=logging.INFO)


def obs_to_policy_format(obs, device='cuda'):
    """Convert observation from environment format to policy expected format."""
    import torch
    
    policy_obs = {}
    
    # Convert agent_pos to observation.state
    if 'agent_pos' in obs:
        policy_obs['observation.state'] = torch.from_numpy(obs['agent_pos']).float().unsqueeze(0).to(device)
    
    # Convert pixels.front and pixels.wrist to observation.images.front and observation.images.wrist
    if 'pixels' in obs:
        if 'front' in obs['pixels']:
            # Convert from [H, W, C] to [C, H, W] format and add batch dimension
            img_front = torch.from_numpy(obs['pixels']['front']).float().permute(2, 0, 1).unsqueeze(0)
            policy_obs['observation.images.front'] = img_front.to(device)
        if 'wrist' in obs['pixels']:
            # Convert from [H, W, C] to [C, H, W] format and add batch dimension  
            img_wrist = torch.from_numpy(obs['pixels']['wrist']).float().permute(2, 0, 1).unsqueeze(0)
            policy_obs['observation.images.wrist'] = img_wrist.to(device)
    
    return policy_obs


def eval_policy(env, policy, n_episodes):
    sum_reward_episode = []
    device = policy.config.device if hasattr(policy.config, 'device') else 'cuda'
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        while True:
            # Convert observation to policy format
            policy_obs = obs_to_policy_format(obs, device)
            action = policy.select_action(policy_obs)
            # Convert action tensor to CPU numpy array
            action_cpu = action.cpu().numpy().squeeze()
            obs, reward, terminated, truncated, _ = env.step(action_cpu)
            episode_reward += reward
            if terminated or truncated:
                break
        sum_reward_episode.append(episode_reward)

    logging.info(f"Success after 20 steps {sum_reward_episode}")
    logging.info(f"success rate {sum(sum_reward_episode) / len(sum_reward_episode)}")


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    env_cfg = cfg.env
    env, teleop_device = make_robot_env(env_cfg)
    dataset_cfg = cfg.dataset
    dataset = LeRobotDataset(repo_id=dataset_cfg.repo_id)
    dataset_meta = dataset.meta

    policy = make_policy(
        cfg=cfg.policy,
        # env_cfg=cfg.env,
        ds_meta=dataset_meta,
    )
    policy.eval()

    eval_policy(env, policy=policy, n_episodes=10)


if __name__ == "__main__":
    main()
