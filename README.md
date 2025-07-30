# AMP-Based Motion Retargeting for Humanoid Robots

This project is based on the Isaac Lab framework and utilizes Adversarial Motion Prior (AMP) for motion retargeting to humanoid robots. AMP enables the imitation of complex human motions using a discriminator-guided reinforcement learning approach. Our system supports retargeting from motion capture datasets to a G1 humanoid robot simulated in Isaac Lab.

## Environment Setup

First, follow the official Isaac Lab installation guide:
[https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

Then, install the AMP Direct module in editable mode. First, activate your Isaac Lab Conda environment, then navigate to the source directory and run:
```bash
conda activate env_isaaclab
cd ~/amp_direct/source/amp_direct
pip install -e .
```

## Dataset Preparation

We use the LAFAN1 Retargeting Dataset (CSV format), available at:
[https://huggingface.co/datasets/lvhaidong/LAFAN1\_Retargeting\_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)

## Data Processing Workflow

1. **Convert CSV to NPZ**

   ```bash
   python data_convert.py
   ```
   The output is `g1.npz` by default.

2. **Check Motion Velocity**

   ```bash
   python test_velocity.py
   ```

   This script reports the mean, maximum linear, and angular velocities.

3. **Clip Excessive Velocities**
   If the motion is too fast and affects training stability:

   ```bash
   python clip_velocity.py
   ```

   The output will be `G1_dance1_clipped.npz`.

4. **Split Long Motions**
   If the motion file is too long:

   ```bash
   python split_motion.py
   ```

   This splits the motion into three parts. You can choose a segment for training.

## Training

> All commands below are executed within the `env_isaaclab` conda environment.

Two motion examples are provided: `G1_fight_clipped_part002.npz` and `G1_fallAndGetUp.npz`.

---

### Training with `G1_fight_clipped_part002.npz`

#### Step 1: Configure motion file

Edit `g1_amp_env_cfg.py`:

```python
motion_file = os.path.join(MOTIONS_DIR, "G1_fight_clipped_part002.npz")
episode_length_s = 10
```

#### Step 2: Train

```bash
python scripts/skrl/train.py --task=Isaac-G1-AMP-Dance-Direct-v0 --headless
```

#### Step 3: Inference (GUI)

```bash
python scripts/skrl/play.py --task=Isaac-G1-AMP-Dance-Direct-v0
```

#### Step 4: Inference (headless + video)

```bash
python scripts/skrl/play.py --task=Isaac-G1-AMP-Dance-Direct-v0 --headless --video --video_length 600
```

#### Step 5: Inference with Pretrained Policy

```bash
python scripts/skrl/play.py --task=Isaac-G1-AMP-Dance-Direct-v0 \
  --checkpoint logs/skrl/g1_amp_dance/2025-07-21_18-07-11_ppo_torch/checkpoints/agent_100000.pt \
  --num_envs 36
```

> Make sure to pull the checkpoint using Git LFS.

---

### Training with `G1_fallAndGetUp.npz` and Soft-DTW reward

#### Step 1: Configure motion file

```python
motion_file = os.path.join(MOTIONS_DIR, "G1_fallAndGetUp.npz")
```

#### Step 2: Optionally tune reward weight

```python
rew_soft_dtw = 2.0
```

#### Step 3: Modify AMP code to enable Soft-DTW

- Line 17:

```python
from .soft_dtw_torch import SoftDTWBatch
```

- Line 35:

```python
self.soft_dtw = SoftDTWBatch(gamma=0.05, normalize=True)
```

- Line 155–156:

```python
soft_dtw_loss = self.soft_dtw(amp_policy_motion, amp_ref_motion)  # [N]
rew_soft_dtw = self.cfg.rew_soft_dtw * torch.exp(-soft_dtw_loss)
```

- Line 184 (uncomment):

```python
imitation_reward = rew_joint_pos + rew_joint_vel + rew_pos + rew_rot + rew_soft_dtw
```

- Line 181 (comment out):

```python
# imitation_reward = rew_joint_pos + rew_joint_vel + rew_pos + rew_rot
```

#### Step 4: Train

```bash
python scripts/skrl/train.py --task=Isaac-G1-AMP-Dance-Direct-v0 --headless
```

#### Step 5: Inference with Pretrained Policy

```bash
python scripts/skrl/play.py --task=Isaac-G1-AMP-Dance-Direct-v0 \
  --checkpoint logs/skrl/g1_amp_dance/2025-07-28_12-09-26_ppo_torch/checkpoints/agent_100000.pt \
  --num_envs 36
```

> Git LFS is required to download pretrained weights.

---

With this setup, you can train non-periodic motions, and enhance trajectory imitation quality using the Soft-DTW reward.
## References

This project is inspired by:  
[https://github.com/linden713/humanoid_amp](https://github.com/linden713/humanoid_amp)
