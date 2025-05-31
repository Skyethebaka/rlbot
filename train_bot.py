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

# Try to import the 220-entry RLViser lookup table (if installed).
try:
    from rlviser_py.tables import RLViser220
except ModuleNotFoundError:
    try:
        from rlviser_py.lookup_tables import RLViser220
    except ModuleNotFoundError:
        RLViser220 = None


# ---------------- Reward Definitions ----------------

class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        out: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            # If the car is on orange team, physics are “flipped”
            cphy = car.physics if car.is_orange else car.inverted_physics
            bphy = state.ball if car.is_orange else state.inverted_ball
            diff = bphy.position - cphy.position
            vel = cphy.linear_velocity
            # dot(velocity, normalized direction_to_ball)
            spd = float(np.dot(vel, diff / (np.linalg.norm(diff) + 1e-6)))
            out[a] = max(spd / common_values.CAR_MAX_SPEED, 0.0)
        return out


class InAirReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        return {a: float(not state.cars[a].on_ground) for a in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        out: Dict[AgentID, float] = {}
        for a in agents:
            # For orange: goal_dir = +1 (ball moving toward ORANGE goal at negative Y),
            # for blue: goal_dir = -1
            goal_dir = 1 if state.cars[a].is_orange else -1
            vely = float(state.ball.linear_velocity[1])
            out[a] = max(goal_dir * vely / common_values.BALL_MAX_SPEED, 0.0)
        return out


# ---------------- Environment Builder ----------------

def build_env(spawn_opponents: bool = True, tick_skip: int = 8):
    """
    Build a single RLGym environment with:
      - 1v1 vs bots if spawn_opponents=True (else 1v0 self-play).
      - tick_skip frames between action decisions.
      - RLViser lookup table if available.
    """
    # Action parser: RLViser220 lookup if installed, else default 90-move table
    if RLViser220 is not None:
        lp = LookupTableAction(custom_table=RLViser220)
    else:
        lp = LookupTableAction()
    ap = RepeatAction(lp, repeats=tick_skip)

    # Termination/truncation conditions
    trunc_cond = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=30),
        TimeoutCondition(timeout_seconds=300)
    )

    # Reward function: combine touch, in‐air, speed‐toward‐ball, ball‐to‐goal, and goal
    reward_fn = CombinedReward(
        (TouchReward(),               0.05),
        (InAirReward(),               0.002),
        (SpeedTowardBallReward(),     0.01),
        (VelocityBallToGoalReward(),  0.1),
        (GoalReward(),                10.0)
    )

    # Observation builder: normalized positions, angles, velocities, boost
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

    # Fix team size (1v1 or 1v0) and apply random kickoff
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=(1 if spawn_opponents else 0)),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=ap,
        reward_fn=reward_fn,
        termination_cond=GoalCondition(),
        truncation_cond=trunc_cond,
        transition_engine=RocketSimEngine()
    )

    return RLGymV2GymWrapper(rlgym_env)


# ---------------- Main Training Script ----------------

if __name__ == "__main__":
    # 1) Determine device (prefer GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) Choose how many parallel envs (n_proc). On 8 H100s, we aim ~24 envs/GPU → 192 total
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        n_proc = 24 * num_gpus
        # Don’t exceed CPU cores
        n_proc = min(n_proc, os.cpu_count())
    else:
        n_proc = max(1, os.cpu_count() - 1)

    # 3) Set PPO hyperparameters smaller on GPU to avoid OOM
    if device == "cuda":
        # On 8×H100:
        #   • network widths halved from [4096,4096,2048,2048] → [2048,2048,1024,1024]
        #   • batch sizes halved from 400k→200k; minibatches 50k; buffer 600k
        policy_layers = [2048, 2048, 1024, 1024]
        critic_layers = [2048, 2048, 1024, 1024]
        ppo_batch_size = 200_000
        ppo_minibatch_size = 50_000
        ts_per_iteration = 500_000  # collect 500k timesteps per PPO update
        exp_buffer_size = 500_000 * 3  # 1.5M total stored transitions
        save_every_ts = 1_000_000
    else:
        # On CPU‐only: keep them smaller still
        policy_layers = [1024, 1024, 512, 512]
        critic_layers = [1024, 1024, 512, 512]
        ppo_batch_size = 100_000
        ppo_minibatch_size = 25_000
        ts_per_iteration = 200_000
        exp_buffer_size = 200_000 * 3
        save_every_ts = 200_000

    learner = Learner(
        build_env,
        n_proc=n_proc,
        min_inference_size=int(n_proc * 0.9),
        device=device,
        ppo_batch_size=ppo_batch_size,
        ppo_minibatch_size=ppo_minibatch_size,
        policy_layer_sizes=policy_layers,
        critic_layer_sizes=critic_layers,
        ts_per_iteration=ts_per_iteration,
        exp_buffer_size=exp_buffer_size,
        ppo_epochs=2,
        ppo_ent_coef=0.01,
        policy_lr=5e-5,
        critic_lr=5e-5,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=save_every_ts,
        timestep_limit=2_000_000_000,
        log_to_wandb=False
    )

    # 4) If multiple GPUs, wrap each network’s internal module in DataParallel
    if device == "cuda" and torch.cuda.device_count() > 1:
        from torch.nn import DataParallel

        ppo_obj = learner.ppo_learner

        # Wrap policy network’s inner model (“.model” or “.net”) 
        if hasattr(ppo_obj, "policy"):
            pol = ppo_obj.policy
            if hasattr(pol, "net"):
                pol.net = DataParallel(pol.net)
            elif hasattr(pol, "model"):
                pol.model = DataParallel(pol.model)
            else:
                raise RuntimeError(
                    "Policy network not found under .net or .model; cannot parallelize."
                )
        else:
            raise RuntimeError("ppo_learner has no attribute 'policy'")

        # Wrap critic network’s inner model
        if hasattr(ppo_obj, "critic"):
            cri = ppo_obj.critic
            if hasattr(cri, "net"):
                cri.net = DataParallel(cri.net)
            elif hasattr(cri, "model"):
                cri.model = DataParallel(cri.model)
            else:
                if hasattr(ppo_obj, "critic_net"):
                    ppo_obj.critic_net = DataParallel(ppo_obj.critic_net)
                else:
                    raise RuntimeError(
                        "Critic network not found under .net, .model, or .critic_net; cannot parallelize."
                    )
        else:
            # Check alternate name (some versions use “value_net”)
            if hasattr(ppo_obj, "value_net"):
                ppo_obj.value_net = DataParallel(ppo_obj.value_net)
            else:
                raise RuntimeError("ppo_learner has no attribute 'critic' or 'value_net'.")

        print(
            f"Wrapped policy and critic internals in DataParallel on {torch.cuda.device_count()} GPUs."
        )

    # 5) Finally, train. This single process will use all GPUs via DataParallel.
    learner.learn()
