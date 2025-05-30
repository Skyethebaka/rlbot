import torch, numpy as np
from train_bot import build_env
from rlgym.rocket_league.rlviser import RLViserRenderer

ckpt = "data/checkpoints/latest/PPO_POLICY.pt"
env = build_env(spawn_opponents=False)
env.renderer = RLViserRenderer()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, act_dim)
        )
    def forward(self, x): return self.m(x)

net = MLP().eval()
net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

obs, _ = env.reset()
while True:
    with torch.no_grad():
        logits = net(torch.from_numpy(obs).float().unsqueeze(0))
    action = np.array([int(logits.argmax())], dtype=np.int32)
    obs, _, term, trunc, _ = env.step(action)
    env.render()
    if term or trunc:
        obs, _ = env.reset()
