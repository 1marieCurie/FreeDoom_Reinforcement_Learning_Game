from stable_baselines3 import PPO
from doom_env import DoomEnv
import time

env = DoomEnv(render_visible=True)

model = PPO.load("doom_ppo_v1_3")

obs, _ = env.reset()

step_count = 0
episode_count = 0

while episode_count < 10:  # Test 10 episodes

    action, _ = model.predict(obs, deterministic=True)#ce que l'agent a vraiment appris

    obs, reward, done, truncated, info = env.step(action)

    step_count += 1

    # Ralentir le rendu pour voir l'agent en action
    time.sleep(0.05)

    if step_count % 50 == 0:
        print(f"step: {step_count} | reward: {reward:.2f} | kills: {info['reward_stats']['kill_bonus']}")

    if done:
        print(f"Episode {episode_count + 1} ended -> reset\n")
        episode_count += 1
        step_count = 0
        obs, _ = env.reset()

print("Test completed!")
env.close()