"""Resume PPO with an old-policy action distillation term on original frames."""
from pathlib import Path
import inspect, json, os, sys
import numpy as np
import torch
from torch import nn

PROJECT=Path('/data/gaoli/projects/Mini-Pi-Plus_BeyondMimic')
TRAIN=PROJECT/'scripts/rsl_rl/train.py'
sys.path.insert(0,str(TRAIN.parent))
source=TRAIN.read_text()
anchor='import whole_body_tracking.tasks  # noqa: F401'
if source.count(anchor)!=1: raise RuntimeError('train anchor changed')
patch=r'''
import inspect as _round2_inspect
import textwrap as _round2_textwrap
import json as _round2_json
import numpy as np
from pathlib import Path
from torch import nn
import rsl_rl.algorithms.ppo as _round2_ppo_mod
from rsl_rl.runners.on_policy_runner import OnPolicyRunner as _Round2Runner
from rsl_rl.modules import EmpiricalNormalization as _Round2Norm

_round2_teacher_path = os.environ["ROUND2_TEACHER_CHECKPOINT"]
_round2_motion_path = os.environ["ROUND2_DISTILL_MOTION"]
_round2_lambda = float(os.environ.get("ROUND2_DISTILL_LAMBDA", "0.1"))
_round2_teacher_ckpt = torch.load(_round2_teacher_path, map_location="cpu", weights_only=False)
_round2_teacher = nn.Sequential(nn.Linear(119,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                                nn.Linear(256,128),nn.ELU(),nn.Linear(128,22)).eval()
_round2_teacher.load_state_dict({k[6:]:v for k,v in _round2_teacher_ckpt["model_state_dict"].items() if k.startswith("actor.")})
_round2_teacher_norm = _Round2Norm(shape=[119]).eval()
_round2_teacher_norm.load_state_dict(_round2_teacher_ckpt["obs_norm_state_dict"])
for _p in _round2_teacher.parameters(): _p.requires_grad_(False)
with np.load(_round2_motion_path) as _round2_motion:
    _round2_commands_np = np.concatenate((_round2_motion["joint_pos"][:89], _round2_motion["joint_vel"][:89]), axis=1).astype(np.float32)
_round2_commands = torch.from_numpy(_round2_commands_np)
_round2_original_update = _round2_ppo_mod.PPO.update
_round2_src = _round2_inspect.getsource(_round2_ppo_mod.PPO.update).replace("def update(self):", "def _round2_update(self):", 1)
_round2_src = _round2_src.replace(
    "            # Surrogate loss\n            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))",
    """            # Original-motion action distillation: preserve the old checkpoint's behavior.
            if not hasattr(self, \"_round2_teacher\"):
                self._round2_teacher = _round2_teacher.to(self.device)
                self._round2_teacher_norm = _round2_teacher_norm.to(self.device)
                self._round2_commands = _round2_commands.to(self.device)
                self._round2_lambda = _round2_lambda
                self._round2_distill_sum = 0.0
                self._round2_distill_batches = 0
            if not hasattr(self, "_round2_student_norm"):
                raise RuntimeError("student normalization must be attached by runner.load")
            normalized_obs = obs_batch[:original_batch_size]
            original_obs = normalized_obs * (self._round2_student_norm._std + 1e-2) + self._round2_student_norm._mean
            with torch.no_grad():
                teacher_actions = self._round2_teacher(self._round2_teacher_norm(original_obs))
            original_mask = torch.cdist(original_obs[:, :44], self._round2_commands).min(dim=1).values < 1.0e-3
            if torch.any(original_mask):
                distill_loss = ((mu_batch - teacher_actions)[original_mask] ** 2).mean()
                self._round2_distill_sum += float(distill_loss.detach().cpu())
                self._round2_distill_batches += 1
            else:
                distill_loss = torch.zeros((), device=obs_batch.device)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))""")
assert "distill_loss =" in _round2_src, "PPO loss injection anchor not found"
_round2_src = _round2_src.replace(
    "            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()",
    "            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self._round2_lambda * distill_loss")
assert "self._round2_lambda * distill_loss" in _round2_src
_round2_src = _round2_src.replace('        return loss_dict', """        Path(os.environ[\"ROUND2_DISTILL_STATS\"]).write_text(_round2_json.dumps({
            "lambda": self._round2_lambda,
            "matched_batches": self._round2_distill_batches,
            "mean_mse": self._round2_distill_sum / max(self._round2_distill_batches, 1),
            "teacher": _round2_teacher_path,
        }, indent=2))
        loss_dict["distillation"] = self._round2_distill_sum / max(self._round2_distill_batches, 1)
        return loss_dict""")
_round2_ppo_mod.__dict__.update({
    "_round2_teacher": _round2_teacher,
    "_round2_teacher_norm": _round2_teacher_norm,
    "_round2_commands": _round2_commands,
    "_round2_lambda": _round2_lambda,
    "_round2_teacher_path": _round2_teacher_path,
    "_round2_json": _round2_json, "Path": Path, "os": os,
})
exec(_round2_textwrap.dedent(_round2_src), _round2_ppo_mod.__dict__)
_round2_ppo_mod.PPO.update = _round2_ppo_mod._round2_update
_round2_load = _Round2Runner.load
def _round2_runner_load(self, *a, **kw):
    result = _round2_load(self, *a, **kw)
    self.alg._round2_student_norm = self.obs_normalizer
    return result
_Round2Runner.load = _round2_runner_load
'''
source=source.replace(anchor,patch)
logging_anchor='    if args_cli.log_dir_path:'
assert source.count(logging_anchor)==1
source=source.replace(logging_anchor,'    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))\n'+logging_anchor)
resume_anchor='        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)'
assert source.count(resume_anchor)==1
source=source.replace(resume_anchor,'''        if os.path.isabs(agent_cfg.load_checkpoint):
            resume_path = agent_cfg.load_checkpoint
            if not os.path.isfile(resume_path): raise FileNotFoundError(resume_path)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)''')
os.chdir(PROJECT)
sys.argv[0]=str(TRAIN)
exec(compile(source,str(TRAIN),'exec'),{'__name__':'__main__','__file__':str(TRAIN)})
