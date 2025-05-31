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

# Try to import the 220-entry RLViser lookup table. Fall back if unavailable.
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

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        out: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            # If the car is on orange team, physics are “flipped”
            cphy = car.physics if car.is_orange else car.inverted_physics
            bphy = state.ball if car.is_orange else state.inverted_ball
            diff = bphy.position - cphy.position
            vel = cphy.linear_velocity
            # dot(velocity, direction_to_ball)
            spd = float(np.dot(vel, diff / (np.linalg.norm(diff) + 1e-6)))
            out[a] = max(spd / common_values.CAR_MAX_SPEED, 0.0)
        return out


class InAirReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {a: float(not state.cars[a].on_ground) for a in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        out: Dict[AgentID, float] = {}
        for a in agents:
            # For orange: goal_dir = +1 (ball moving toward orange goal at negative Y),
            # for blue: goal_dir = -1
            goal_dir = 1 if state.cars[a].is_orange else -1
            vely = float(state.ball.linear_velocity[1])
            out[a] = max(goal_dir * vely / common_values.BALL_MAX_SPEED, 0.0)
        return out


# ---------------- Environment Builder ----------------

def build_env(spawn_opponents: bool = True, tick_skip: int = 8):
    """
    Build a single RLGym environment with:
      - 1v1 vs bots if spawn_opponents=True, else 1v0 (self‐play).
      - 8× tick_skip frames between actions.
      - RLViser lookup table if available.
    """
    # Action parser: RLViser220 lookup if installed, else default lookup
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

    # Reward function: combine several shaped rewards
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

    # 2) Choose how many parallel environments (n_proc)
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        # e.g. 24 envs per GPU → 8 GPUs ⇒ 192 envs
        n_proc = 24 * num_gpus
        # Don’t exceed total CPU cores
        n_proc = min(n_proc, os.cpu_count())
    else:
        # Leave one core for OS, use the rest
        n_proc = max(1, os.cpu_count() - 1)

    # 3) Construct the PPO learner. We use larger batch sizes on GPU, smaller on CPU
    learner = Learner(
        build_env,
        n_proc=n_proc,
        min_inference_size=int(n_proc * 0.9),
        device=device,
        # On GPU, we use big batches so each H100 sees ~100k states per update
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

    # 4) If we have multiple GPUs, wrap only the internal network(s) in DataParallel
    if device == "cuda" and torch.cuda.device_count() > 1:
        from torch.nn import DataParallel

        ppo_obj = learner.ppo_learner

        # — Wrap policy’s internal net/model, not the outer policy object —
        if hasattr(ppo_obj, "policy"):
            pol = ppo_obj.policy
            if hasattr(pol, "net"):
                pol.net = DataParallel(pol.net)
            elif hasattr(pol, "model"):
                pol.model = DataParallel(pol.model)
            else:
                # If RLgym-PPO uses a different attribute name, you could add more checks here
                raise RuntimeError("Policy network not found under .net or .model; cannot parallelize.")
        else:
            raise RuntimeError("ppo_learner has no attribute 'policy'")

        # — Wrap critic’s network portion —
        # Common names in RLgym-PPO are “.critic” (outer), inside which might be .net or .model
        if hasattr(ppo_obj, "critic"):
            cri = ppo_obj.critic
            if hasattr(cri, "net"):
                cri.net = DataParallel(cri.net)
            elif hasattr(cri, "model"):
                cri.model = DataParallel(cri.model)
            else:
                # Some versions call it “.critic_net” or so—check those too
                if hasattr(ppo_obj, "critic_net"):
                    ppo_obj.critic_net = DataParallel(ppo_obj.critic_net)
                else:
                    raise RuntimeError("Critic network not found under .net, .model, or .critic_net; cannot parallelize.")
        else:
            # It’s possible your RLgym-PPO version names the critic differently,
            # e.g. “.value_net.” If so, check and wrap that instead:
            if hasattr(ppo_obj, "value_net"):
                ppo_obj.value_net = DataParallel(ppo_obj.value_net)
            else:
                raise RuntimeError("ppo_learner has no attribute 'critic' or 'value_net'.")

        print(f"Wrapped policy and critic internals in DataParallel on {torch.cuda.device_count()} GPUs.")

    # 5) Finally, start training. This one process will use all GPUs for each update.
    learner.learn()
