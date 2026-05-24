import gymnasium as gym
from gymnasium import spaces
from vizdoom import *

import numpy as np
import cv2


class DoomEnv(gym.Env):

    def __init__(self, render_visible=False):

        super().__init__()

        # ==========================================
        # INIT VIZDOOM
        # ==========================================
        self.game = DoomGame()

        self.game.load_config("deathmatch_mine.cfg")

        # fenêtre visible uniquement pour test
        self.game.set_window_visible(render_visible)

        # mode
        if render_visible:

            self.game.set_mode(Mode.PLAYER)

            self.game.set_screen_resolution(
                ScreenResolution.RES_640X480
            )

        else:

            self.game.set_mode(
                Mode.ASYNC_PLAYER
            )

        self.game.init()

        # ==========================================
        # ACTION SPACE
        # ==========================================
        buttons = self.game.get_available_buttons()

        self.button_map = {
            b: i for i, b in enumerate(buttons)
        }

        n = len(buttons)

        def action_with(button_list):

            action = [0] * n

            for b in button_list:

                action[
                    self.button_map[b]
                ] = 1

            return action

        # ==========================================
        # ACTIONS
        # ==========================================
        self.actions = [

            # avancer
            action_with([
                Button.MOVE_FORWARD
            ]),

            # reculer
            action_with([
                Button.MOVE_BACKWARD
            ]),

            # gauche
            action_with([
                Button.MOVE_LEFT
            ]),

            # droite
            action_with([
                Button.MOVE_RIGHT
            ]),

            # tourner gauche
            action_with([
                Button.TURN_LEFT
            ]),

            # tourner droite
            action_with([
                Button.TURN_RIGHT
            ]),

            # tirer
            action_with([
                Button.ATTACK
            ]),

            # avancer + tirer
            action_with([
                Button.MOVE_FORWARD,
                Button.ATTACK
            ]),
        ]

        self.action_space = spaces.Discrete(
            len(self.actions)
        )

        # ==========================================
        # OBSERVATION SPACE
        # ==========================================
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        # ==========================================
        # TRAINING ZONE
        # ==========================================
        self.max_distance = 300

        # ==========================================
        # REWARD STATS
        # ==========================================
        self.reward_stats = {

            "kill_bonus": 0.0,

            "damage_penalty": 0.0,

            "ammo_penalty": 0.0,

            "survival_bonus": 0.0,

            "time_penalty": 0.0,

            "zone_penalty": 0.0,

            "death_penalty": 0.0,
        }

    # ==========================================
    # PREPROCESS
    # ==========================================
    def preprocess(self, img):

        # ViZDoom retourne parfois (C,H,W)
        if (
            len(img.shape) == 3
            and img.shape[0] in [1, 3]
        ):

            img = np.transpose(
                img,
                (1, 2, 0)
            )

        # grayscale -> RGB
        if len(img.shape) == 2:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2RGB
            )

        # sécurité
        if img.shape[2] > 3:

            img = img[:, :, :3]

        # resize
        img = cv2.resize(
            img,
            (84, 84)
        )

        return img.astype(np.uint8)

    # ==========================================
    # RESET
    # ==========================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # ======================================
        # PRINT PREVIOUS EPISODE
        # ======================================
        if hasattr(self, "episode_reward"):

            print(
                "\n========== EPISODE SUMMARY =========="
            )

            print(
                f"Episode reward : "
                f"{self.episode_reward:.3f}"
            )

            print(
                f"Enemy kills    : "
                f"{self.total_kills}"
            )

            print(
                f"Episode steps  : "
                f"{self.step_count}"
            )

            print("\nReward components:")

            for k, v in self.reward_stats.items():

                print(
                    f"  {k:<18}: "
                    f"{v:.3f}"
                )

            print(
                "=====================================\n"
            )

        # ======================================
        # RESET STATS
        # ======================================
        self.reward_stats = {

            "kill_bonus": 0.0,

            "damage_penalty": 0.0,

            "ammo_penalty": 0.0,

            "survival_bonus": 0.0,

            "time_penalty": 0.0,

            "zone_penalty": 0.0,

            "death_penalty": 0.0,
        }

        # ======================================
        # NEW EPISODE
        # ======================================
        self.game.new_episode()

        state = self.game.get_state()

        if state is None:

            obs = np.zeros(
                (84, 84, 3),
                dtype=np.uint8
            )

            return obs, {}

        vars = state.game_variables

        # ======================================
        # INITIAL POSITION
        # ======================================
        self.start_x = vars[5]
        self.start_y = vars[6]

        # ======================================
        # MEMORY
        # ======================================
        self.previous_kills = vars[0]

        self.previous_health = vars[1]

        self.previous_ammo = vars[4]

        self.previous_damage = vars[7]

        # ======================================
        # EPISODE TRACKERS
        # ======================================
        self.total_kills = 0

        self.episode_reward = 0.0

        self.step_count = 0

        obs = self.preprocess(
            state.screen_buffer
        )

        return obs, {}

    # ==========================================
    # STEP
    # ==========================================
    def step(self, action_idx):

        action = self.actions[action_idx]

        # frame skip
        self.game.make_action(action, 4)

        done = self.game.is_episode_finished()

        # ======================================
        # TERMINATED
        # ======================================
        if done:

            obs = np.zeros(
                (84, 84, 3),
                dtype=np.uint8
            )

            death_penalty = -0.3

            self.reward_stats[
                "death_penalty"
            ] += abs(death_penalty)

            self.episode_reward += death_penalty

            return (
                obs,
                death_penalty,
                True,
                False,
                {}
            )

        state = self.game.get_state()

        if state is None:

            obs = np.zeros(
                (84, 84, 3),
                dtype=np.uint8
            )

            return (
                obs,
                -0.3,
                True,
                False,
                {}
            )

        # ======================================
        # OBS
        # ======================================
        obs = self.preprocess(
            state.screen_buffer
        )

        vars = state.game_variables

        # ======================================
        # GAME VARIABLES
        # ======================================
        kills = vars[0]

        health = vars[1]

        ammo = vars[4]

        x = vars[5]
        y = vars[6]

        damage_count = vars[7]

        # ======================================
        # DELTAS
        # ======================================
        damage_taken = (
            damage_count
            - self.previous_damage
        )

        ammo_used = (
            self.previous_ammo
            - ammo
        )

        # ======================================
        # REWARD INIT
        # ======================================
        custom_reward = 0.0

        self.step_count += 1

        # ======================================
        # SURVIVAL BONUS
        # ======================================
        survival_bonus = 0.005

        custom_reward += survival_bonus

        self.reward_stats[
            "survival_bonus"
        ] += survival_bonus

        # ======================================
        # TIME PENALTY
        # ======================================
        time_penalty = -0.001

        custom_reward += time_penalty

        self.reward_stats[
            "time_penalty"
        ] += abs(time_penalty)

        # ======================================
        # DAMAGE PENALTY
        # ======================================
        damage_penalty = 0.0

        if damage_taken > 0:

            damage_penalty = (
                -0.02 * damage_taken
            )

            custom_reward += damage_penalty

            self.reward_stats[
                "damage_penalty"
            ] += abs(damage_penalty)

        # ======================================
        # AMMO PENALTY
        # ======================================
        ammo_penalty = 0.0

        if ammo_used > 0:

            ammo_penalty = (
                -0.003 * ammo_used
            )

            custom_reward += ammo_penalty

            self.reward_stats[
                "ammo_penalty"
            ] += abs(ammo_penalty)

        # ======================================
        # KILL BONUS
        # ======================================
        kill_bonus = 0.0

        if kills > self.previous_kills:

            new_kills = (
                kills
                - self.previous_kills
            )

            kill_bonus = (
                1.0 * new_kills
            )

            custom_reward += kill_bonus

            self.total_kills += new_kills

            self.reward_stats[
                "kill_bonus"
            ] += kill_bonus

        # ======================================
        # LOW HEALTH PENALTY
        # ======================================
        if health < 25:

            low_health_penalty = -0.01

            custom_reward += (
                low_health_penalty
            )

        # ======================================
        # SPATIAL LIMIT
        # ======================================
        distance_x = abs(
            x - self.start_x
        )

        distance_y = abs(
            y - self.start_y
        )

        if (
            distance_x > self.max_distance
            or
            distance_y > self.max_distance
        ):

            zone_penalty = -1.0

            custom_reward += zone_penalty

            self.reward_stats[
                "zone_penalty"
            ] += abs(zone_penalty)

            done = True

        # ======================================
        # TOTAL EPISODE REWARD
        # ======================================
        self.episode_reward += custom_reward

        # ======================================
        # UPDATE MEMORY
        # ======================================
        self.previous_kills = kills

        self.previous_health = health

        self.previous_ammo = ammo

        self.previous_damage = damage_count

        # ======================================
        # INFO
        # ======================================
        info = {

            "reward":
                float(custom_reward),

            "kills":
                int(self.total_kills),

            "health":
                float(health),

            "ammo":
                float(ammo),

            "damage_taken":
                float(damage_taken),

            "ammo_used":
                float(ammo_used),

            "step":
                int(self.step_count),

            "position_x":
                float(x),

            "position_y":
                float(y),

            "reward_stats":
                self.reward_stats.copy(),
        }

        return (
            obs,
            custom_reward,
            done,
            False,
            info
        )

    # ==========================================
    # RENDER
    # ==========================================
    def render(self):
        pass

    # ==========================================
    # CLOSE
    # ==========================================
    def close(self):

        self.game.close()