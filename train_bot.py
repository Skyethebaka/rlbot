#!/usr/bin/env python3
import os
import torch
import numpy as np
from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID, RLGym
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator

from rlgym_ppo.util import RLGymV2GymWrapper
from rlgym_ppo import Learner

try:
    from rlviser_py.tables import RLViser220
except ModuleNotFoundError:
    try:
        from rlviser_py.lookup_tables import RLViser220
    except ModuleNotFoundError:
        RLViser220 = None


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None:
        pass

    def get_rewards(self, agents, state, *_) -> Dict[AgentID, float]:
        out: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            cphy = car.physics if car.is_orange else car.inverted_physics
            bphy = state.ball if car.is_orange else state.inverted_ball
            diff = bphy.position - cphy.position
            vel = cphy.linear_velocity
            spd = np.dot(vel, diff / (np.linalg.norm(diff) + 1e-6))
            out[a] = max(spd / common_values.CAR_MAX_SPEED, 0.0)
        return out


class InAirReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None:
        pass

    def get_rewards(self, agents, state, *_) -> Dict[AgentID, float]:
        return {a: float(not state.cars[a].on_ground) for a in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None:
        pass

    def get_rewards(self, agents, state, *_) -> Dict[AgentID, float]:
        out: Dict[AgentID, float] = {}
        for a in agents:
            # If Orange, goal is at negative Y; if Blue, goal is at +Y
            goal_dir = 1 if state.cars[a].is_orange else -1
            vely = state.ball.linear_velocity[1]
            out[a] = max(goal_dir * vely / common_values.BALL_MAX_SPEED, 0.0)
        return out


def build_env(spawn_opponents: bool = True, tick_skip: int = 8):
    lp = LookupTableAction(custom_table=RLViser220) if RLViser220 else LookupTableAction()
    ap = RepeatAction(lp, repeats=tick_skip)

    reward_fn = CombinedReward(
        (TouchReward(), 0.05),
        (InAirReward(), 0.002),
        (SpeedTowardBallReward(), 0.01),
        (VelocityBallToGoalReward(), 0.1),
        (GoalReward(), 10.0)
    )

    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([
            1 / common_values.SIDE_WALL_X,
            1 / common_values.BACK_NET_Y,
            1 / common_values.CEILING_Z
        ]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=0.01
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=ap,
        reward_fn=reward_fn,
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(30),
            TimeoutCondition(300)
        ),
        transition_engine=RocketSimEngine()
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    # 1) Device selection: prefer "cuda" if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) Set n_proc based on CPU vs GPU environment
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        # We'll assign 24 envs per GPU (for 8 GPUs → 192 envs)
        n_proc = 24 * num_gpus
        # But do not ever exceed the number of vCPUs available
        if n_proc > os.cpu_count():
            n_proc = os.cpu_count()
    else:
        # On CPU-only, use (cpu_count - 1) to leave one core for OS
        n_proc = max(1, os.cpu_count() - 1)

    # 3) Create the RLgym-PPO Learner
    learner = Learner(
        build_env,
        n_proc=n_proc,
        min_inference_size=int(n_proc * 0.9),
        device=device,
        ppo_batch_size=(400_000 if device == "cuda" else 100_000),
        ppo_minibatch_size=(100_000 if device == "cuda" else 25_000),
        policy_layer_sizes=([4096, 4096, 2048, 2048] if device == "cuda" else [1024, 1024, 512, 512]),
        critic_layer_sizes=([4096, 4096, 2048, 2048] if device == "cuda" else [1024, 1024, 512, 512]),
        ts_per_iteration=(1_000_000 if device == "cuda" else 200_000),
        exp_buffer_size=(400_000 * 3 if device == "cuda" else 100_000 * 3),
        ppo_epochs=2,
        ppo_ent_coef=0.01,
        policy_lr=5e-5,
        critic_lr=5e-5,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=(1_000_000 if device == "cuda" else 200_000),
        timestep_limit=2_000_000_000,
        log_to_wandb=False
    )

    # 4) Wrap policy & critic in DataParallel, but *only if* those attributes exist
    if device == "cuda" and torch.cuda.device_count() > 1:
        from torch.nn import DataParallel

        # The PPOLearner instance lives at learner.ppo_learner
        ppo_obj = learner.ppo_learner

        # Wrap the policy network if it exists
        if hasattr(ppo_obj, "policy"):
            ppo_obj.policy = DataParallel(ppo_obj.policy)
        else:
            # In case RLgym-PPO renamed it—try common alternatives:
            if hasattr(ppo_obj, "actor"):
                ppo_obj.actor = DataParallel(ppo_obj.actor)

        # Wrap the critic (value network) if it exists
        if hasattr(ppo_obj, "critic"):
            ppo_obj.critic = DataParallel(ppo_obj.critic)
        else:
            # Maybe it’s called "critic_net" or "value_net"
            if hasattr(ppo_obj, "critic_net"):
                ppo_obj.critic_net = DataParallel(ppo_obj.critic_net)
            elif hasattr(ppo_obj, "value_net"):
                ppo_obj.value_net = DataParallel(ppo_obj.value_net)

    # 5) Start training (this one process will drive all GPUs if DataParallel succeeded)
    learner.learn()
