# 机器人模仿学习比赛指南

## 概述

本指南将带您完成完整的机器人模仿学习流程：从数据采集、模型训练到策略推理。本指南基于LeRobot框架，使用仿真环境进行机器人抓取任务的模仿学习。

## 环境准备

### 1. 系统要求
- Ubuntu 20.04+ 或类似Linux发行版
- Python 3.10
- CUDA支持的GPU（推荐）
- 至少8GB内存

⚠️注意：经过验证测试的显卡有 Nvidia的40系列（推荐），30系列。  
不支持50系列。

### 2. 安装Miniconda
```bash
# 下载并安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 3. 创建虚拟环境
```bash
# 创建Python 3.12环境
conda create -y -n dora_lerobot python=3.12

# 激活环境
conda activate dora_lerobot

# 安装ffmpeg
conda install ffmpeg -c conda-forge
```

### 4. 安装Dora Lerobot

⚠️注意：仍然是在 `conda` 的 `dora_lerobot` 环境下面操作

```bash
# 克隆仓库
git clone https://github.com/DoraCN/dora-lerobot.git
cd dora-lerobot

# 安装依赖
pip install -e ".[hilserl]"
```

## 完整工作流程

### 第一阶段：数据采集

#### 1. 创建数据采集配置
创建文件 `env_config_gym_hil_il.json`：

```json
{
  "env": {
    "name": "gym_hil",
    "task": "PandaPickCubeKeyboard-v0",
    "fps": 30,
    "processor": {
      "reset": {
        "control_time_s": 60.0, # 采集一组数据的时间长度，经过测试下来60秒时间适合
        "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
        "reset_time_s": 2.0, # 重置环境的时间
        "terminate_on_success": false  # 是否运行终端结束，一般默认 false
      }
    }
  },
  "dataset": {
    "repo_id": "dora_lerobot/gym_hil",
    "task": "pick_cube",
    "num_episodes_to_record": 10, # 采集的全部组数，经过测试至少10组开始效果普遍会好起来
    "replay_episode": null,
    "push_to_hub": false
  },
  "mode": "record",
  "device": "cuda"
}
```
其中修改 `"repo_id"`: `"dora_lerobot/gym_hil"`,为您本地的路径(`此路径必须是本地没有，会自动创建。若存在会停止采集`)，通常本地会存储在 `$HOME/.cache/huggingface/lerobot/dora_lerobot/gym_hil` 下。

#### 2. 开始数据采集
```bash
# 激活环境
conda activate dora_lerobot

# 运行数据采集
python -m lerobot.rl.gym_manipulator --config_path env_config_gym_hil_il.json 
```

**键盘控制说明：**

  - `方向键`：控制机械臂末端在X和Y轴上水平前后左右运动
  - `左Shift`: Z轴向下
  - `右Shift`: Z轴向上
  - `左Ctrl`: 夹爪张开
  - `右Ctrl`: 夹爪关闭
  - `Enter`: 结束当前阶段的采集任务，并标记为成功状态
  - `Backspace`: 结束当前阶段的采集任务，并标记为失败状态
  - `空格键`: 开始/暂停当前状态
  - `ESC`: 退出整个采集任务

> ⚠️注意：    
> **开始采集数据时，键盘控制之前必须通过按下 `空格键` 开始，否则仿真环境不响应任何按键。**

⚠️此时遇到的常见问题：

```bash
# 问题一：
ibEGL warning: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)    

#解决方法：
sudo mkdir /usr/lib/dri
sudo ln -sf /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so   dri/swrast_dri.so

# 问题二：
libGL error: MESA-LOADER: failed to open swrast: /home/dora/miniconda3/envs/lerobot_v3/bin/../lib/libstdc++.so.6: version GLIBCXX_3.4.30 not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)    

#解决方法
sudo ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6   /home/dora/miniconda3/envs/当前的conda环境名称/lib/libstdc++.so.6
```

**采集建议：**
- 至少采集10个完整的episode
- 每个episode包含完整的抓取-放置任务
- 保持动作流畅，避免突然的急停
- 确保任务成功完成（绿色提示）

### 第二阶段：模型训练

#### 1. 训练ACT策略
```bash
# 激活环境
conda activate dora_lerobot

# 开始训练
lerobot-train \
  --dataset.repo_id=dora_lerobot/gym_hil \
  --policy.type=act \
  --output_dir=outputs/train/il_gym \
  --job_name=il_gym \
  --policy.device=cuda \
  --steps=5000 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --wandb.enable=false \
  --policy.push_to_hub=false
```

**训练参数说明：**
- `--dataset.repo_id=dora_lerobot/gym_hil`：使用您采集的数据集
- `--policy.type=act`：使用ACT（Action Chunking with Transformers）算法
- `--output_dir=outputs/train/il_gym`：存放训练好的模型的地址
- `--steps=5000`：总训练步数
- `--save_freq=1000`：多少步保存一个检查点
- `--wandb.enable=false`: 不开启wandb
- `--policy.push_to_hub=false`：不自动上传至huggingface仓库，保存在本地

**训练过程：**
- 训练会自动保存checkpoint到 `outputs/train/il_gym/checkpoints/`
- 推荐训练至少5000步
- 训练完成后会生成 `pretrained_model` 文件夹

### 第三阶段：策略推理

#### 1. 创建推理配置
创建文件 `eval_config_minimal.json`：

```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "gym_hil",
    "task": "PandaPickCubeKeyboard-v0",
    "fps": 30,
    "robot": null,
    "teleop": null,
    "processor": {
      "control_mode": "keyboard",
      "observation": {
        "add_joint_velocity_to_observation": true,
        "add_current_to_observation": true,
        "display_cameras": false
      },
      "image_preprocessing": {
        "crop_params_dict": {
          "observation.images.front": [0, 0, 128, 128],
          "observation.images.wrist": [0, 0, 128, 128]
        },
        "resize_size": [128, 128]
      },
      "gripper": {
        "use_gripper": true,
        "gripper_penalty": 0
      },
      "reset": {
        "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
        "reset_time_s": 2.0,
        "control_time_s": 20.0,
        "terminate_on_success": true
      },
      "reward_classifier": {
        "pretrained_path": null
      }
    }
  },
  "dataset": {
    "repo_id": "dora_lerobot/gym_hil",
    "use_imagenet_stats": false
  },
  "policy": {
    "type": "act",
    "device": "cuda",
    "n_obs_steps": 1,
    "normalization_mapping": {
      "VISUAL": "MEAN_STD",
      "STATE": "MEAN_STD",
      "ACTION": "MEAN_STD"
    },
    "input_features": {
      "agent_pos": {
        "type": "STATE",
        "shape": [18]
      },
      "pixels.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "pixels.wrist": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      }
    },
    "output_features": {
      "action": {
        "type": "ACTION",
        "shape": [4]
      }
    },
    "chunk_size": 100,
    "n_action_steps": 100,
    "vision_backbone": "resnet18",
    "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
    "replace_final_stride_with_dilation": false,
    "pre_norm": false,
    "dim_model": 512,
    "n_heads": 8,
    "dim_feedforward": 3200,
    "feedforward_activation": "relu",
    "n_encoder_layers": 4,
    "n_decoder_layers": 1,
    "use_vae": true,
    "latent_dim": 32,
    "n_vae_encoder_layers": 4,
    "temporal_ensemble_coeff": null,
    "dropout": 0.1,
    "kl_weight": 10.0,
    "optimizer_lr": 1e-05,
    "optimizer_weight_decay": 0.0001,
    "optimizer_lr_backbone": 1e-05,
    "use_amp": false
  }
}
```

#### 2. 运行策略推理
```bash
# 激活环境
conda activate dora_lerobot

# 运行推理
python -m lerobot.rl.eval_policy --config_path=eval_config_minimal.json --policy.path="outputs/train/il_gym/checkpoints/005000/pretrained_model"
```

**注意：** 请将路径中的 `005000` 替换为您实际训练的checkpoint步数。

## 高级配置选项

### 数据采集参数调优

#### 图像预处理参数
```json
"image_preprocessing": {
  "crop_params_dict": {
    "observation.images.front": [x, y, width, height],
    "observation.images.wrist": [x, y, width, height]
  },
  "resize_size": [width, height]
}
```

#### 控制参数
```json
"processor": {
  "control_mode": "keyboard",  // 或 "gamepad"
  "observation": {
    "add_joint_velocity_to_observation": true,  // 添加关节速度
    "add_current_to_observation": true,        // 添加电流信息
    "display_cameras": false                   // 是否显示相机画面
  }
}
```

### 训练参数调优

#### ACT策略参数
```bash
# 基础训练命令
lerobot-train --dataset.repo_id=dora_lerobot/gym_hil --policy.type=act

# 高级参数示例
lerobot-train \
  --dataset.repo_id=dora_lerobot/gym_hil \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.dim_model=256 \
  --policy.n_heads=4 \
  --policy.n_encoder_layers=2 \
  --policy.dropout=0.1 \
  --policy.optimizer_lr=1e-4 \
  --policy.use_vae=true \
  --policy.latent_dim=16
```

#### 训练监控
```bash
# 使用Weights & Biases监控训练
wandb login
lerobot-train --dataset.repo_id=dora_lerobot/gym_hil --policy.type=act --wandb.project=my_robot_project
```

### 推理参数调优

#### 设备选择
```json
"policy": {
  "device": "cuda",    // 使用GPU
  // "device": "cpu",  // 使用CPU（较慢）
}
```

#### 观察步数调整
```json
"policy": {
  "n_obs_steps": 1,    // 使用1步历史
  // "n_obs_steps": 3, // 使用3步历史（需要更多内存）
}
```

## 故障排除

### 常见错误及解决方案

#### 1. CUDA内存不足
```bash
# 减少batch size或使用CPU
"policy": {
  "device": "cpu"
}
```

#### 2. 数据集加载失败
```bash
# 检查数据集是否存在
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('dora_lerobot/gym_hil'); print(ds.meta)"
```

#### 3. 模型加载失败
```bash
# 检查checkpoint路径
ls -la outputs/train/il_gym/checkpoints/
```

#### 4. 环境启动失败
```bash
# 检查MuJoCo安装
python -c "import mujoco; print('MuJoCo installed successfully')"
```

### 性能优化建议

#### 1. 数据采集优化
- 使用SSD存储数据集
- 确保充足的磁盘空间（至少10GB）
- 采集时关闭不必要的程序

#### 2. 训练优化
- 使用GPU进行训练
- 调整batch size以充分利用GPU内存
- 使用混合精度训练（`use_amp: true`）

#### 3. 推理优化
- 使用GPU进行推理
- 减少不必要的观察历史
- 优化图像预处理参数

## 评估指标

### 任务成功率
- 成功抓取并放置物体
- 任务完成时间
- 动作流畅度

### 模型性能
- 训练损失收敛情况
- 验证集性能
- 推理速度

## 提交要求

### 必需文件
1. 训练好的模型checkpoint
2. 数据集配置文件
3. 推理配置文件
4. 训练日志

### 评估标准
1. **任务完成率**：成功完成抓取任务的比例
2. **效率**：完成任务所需的时间
3. **稳定性**：多次运行的一致性
4. **创新性**：算法改进或参数调优

## 技术支持

### 官方资源
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot HuggingFace](https://huggingface.co/lerobot)
- [Discord社区](https://discord.gg/s3KuuzsPFb)

### 常见问题
1. **Q: 如何增加训练数据？**
   A: 采集更多episode，建议至少10个episode

2. **Q: 如何提高成功率？**
   A: 优化数据质量，调整训练参数，增加训练步数

3. **Q: 如何加速训练？**
   A: 使用GPU，调整batch size，使用混合精度

4. **Q: 如何处理不同任务？**
   A: 修改环境配置中的task参数，重新采集数据

## 总结

本指南提供了完整的机器人模仿学习流程，从数据采集到模型训练再到策略推理。通过遵循这些步骤，您可以成功训练出能够执行抓取任务的机器人策略。

记住：
- 数据质量决定模型性能
- 充分的训练时间很重要
- 参数调优需要耐心和实验
- 评估和迭代是改进的关键

祝您在比赛中取得好成绩！🎯
