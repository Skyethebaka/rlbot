import os, torch, numpy as np
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
    def reset(self, *_) -> None: ...
    def get_rewards(self, agents, state, *_):
        out: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            cphy = car.physics if car.is_orange else car.inverted_physics
            bphy = state.ball if car.is_orange else state.inverted_ball
            diff = bphy.position - cphy.position
            vel = cphy.linear_velocity
            spd = np.dot(vel, diff / (np.linalg.norm(diff) + 1e-6))
            out[a] = max(spd / common_values.CAR_MAX_SPEED, 0.)
        return out

class InAirReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None: ...
    def get_rewards(self, agents, state, *_):
        return {a: float(not state.cars[a].on_ground) for a in agents}

class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, *_) -> None: ...
    def get_rewards(self, agents, state, *_):
        out: Dict[AgentID, float] = {}
        for a in agents:
            goal_dir = 1 if state.cars[a].is_orange else -1
            vely = state.ball.linear_velocity[1]
            out[a] = max(goal_dir * vely / common_values.BALL_MAX_SPEED, 0.)
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
    env = RLGym(
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
    return RLGymV2GymWrapper(env)

if __name__ == "__main__":
    device = "cuda"
    n_proc = 192
    learner = Learner(
        build_env,
        n_proc=n_proc,
        min_inference_size=int(n_proc * 0.9),
        device=device,
        ppo_batch_size=800_000,
        ppo_minibatch_size=200_000,
        policy_layer_sizes=[4096, 4096, 2048, 2048],
        critic_layer_sizes=[4096, 4096, 2048, 2048],
        ts_per_iteration=2_000_000,
        exp_buffer_size=2_000_000,
        ppo_epochs=2,
        ppo_ent_coef=0.01,
        policy_lr=5e-5,
        critic_lr=5e-5,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=2_000_000,
        timestep_limit=2_000_000_000,
        log_to_wandb=False
    )
    learner.learn()
