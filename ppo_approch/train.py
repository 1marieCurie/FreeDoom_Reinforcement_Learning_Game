from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from doom_env import DoomEnv
from stable_baselines3.common.callbacks import CheckpointCallback


env = DoomEnv()

env = Monitor(env)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="doom_model_v1_3"
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="./ppo_doom_tensorboard/"
)

model.learn(
    total_timesteps=300000, # 300k de timestemps
    tb_log_name="deathmatch_test",
    callback=checkpoint_callback
)


model.save("doom_ppo_v1_3")