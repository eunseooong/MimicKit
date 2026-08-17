import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.ppo_agent as ppo_agent
import learning.rl_util as rl_util
import learning.style_imp_model as style_imp_model
import util.mp_util as mp_util
import util.torch_util as torch_util

class StyleImpAgent(ppo_agent.PPOAgent):
    """PPO over a policy whose pd targets are shared across styles and whose impedance is
    conditioned on the style. See StyleImpModel for the architecture."""

    def _load_params(self, config):
        super()._load_params(config)
        # only meaningful when the pose head sees the full obs, see StyleImpModel
        self._pose_style_reg_weight = config.get("pose_style_reg_weight", 0.0)
        self._pose_style_reg_bins = config.get("pose_style_reg_bins", 20)
        self._pose_style_reg_min_count = config.get("pose_style_reg_min_count", 2)
        self._pose_style_reg_normalize = config.get("pose_style_reg_normalize", True)
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = style_imp_model.StyleImpModel(model_config, self._env)
        return

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        style = self._env.get_style_onehot()
        norm_action_dist = self._model.eval_actor(norm_obs, style)

        if (self._mode == base_agent.AgentMode.TRAIN):
            norm_a_rand = norm_action_dist.sample()
            norm_a_mode = norm_action_dist.mode

            exp_prob = self._get_exp_prob()
            exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob, device=self._device, dtype=torch.float)
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)

        elif (self._mode == base_agent.AgentMode.TEST):
            norm_a = norm_action_dist.mode
            rand_action_mask = torch.zeros_like(norm_a[..., 0])

        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)

        norm_a_logp = norm_action_dist.log_prob(norm_a)

        norm_a = norm_a.detach()
        norm_a_logp = norm_a_logp.detach()
        a = self._a_norm.unnormalize(norm_a)

        a_info = {
            "a_logp": norm_a_logp,
            "rand_action_mask": rand_action_mask,
            "style": style,
            "phase": self._env.get_motion_phase()
        }
        return a, a_info

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("style", action_info["style"])
        self._exp_buffer.record("phase", action_info["phase"])
        return

    def _compute_pose_style_reg(self, norm_obs, style, phase):
        """Soft stand-in for restricting the pose head to the phase: penalizes the disagreement
        between the pd targets of different styles at the same point in the motion.

        Normalizing by the spread of the targets over the whole motion keeps wide-range joints from
        dominating, puts the loss on the same [0, ~1] scale as the reported pose_style_ratio, and
        so lets the weight carry over between motions and configs.

        It does NOT rule out the degenerate minimum. Collapsing q* to a constant zeroes the
        numerator and the denominator together, and the ratio goes to zero just like the raw
        variance does, so both forms reward a policy that stops moving. Only the tracking reward
        pushes back on that; watch pose_var in the diagnostics to catch it.
        """
        pose = self._model.eval_pose(norm_obs, style)
        style_var = self._calc_pose_style_var(pose, style, phase)
        if (style_var is None):
            return None

        if (self._pose_style_reg_normalize):
            total_var = torch.var(pose, dim=0).detach()
            return torch.mean(style_var / torch.clamp_min(total_var, 1e-6))
        return torch.sum(style_var)

    def _calc_pose_style_var(self, pose, style, phase):
        """Per-dim cross-style variance of the pd targets within a phase bin, averaged over bins.
        Returns None when no bin holds at least two styles. Shape is (num_pose_dims,).

        The styles are assumed phase aligned, so samples in the same phase bin are at the same
        point of the motion regardless of style, and any disagreement between their pd targets is
        style information the policy should not be using. Binning by phase is essential: comparing
        per-style means over the whole motion averages the leak away and reports a healthy number
        for a broken policy.
        """
        num_bins = self._pose_style_reg_bins
        num_styles = self._env.get_num_styles()

        style_ids = torch.argmax(style, dim=-1)
        bin_ids = torch.clamp((phase * num_bins).type(torch.long), 0, num_bins - 1)
        cell_ids = bin_ids * num_styles + style_ids

        num_cells = num_bins * num_styles
        counts = torch.zeros([num_cells], device=self._device, dtype=pose.dtype)
        counts.index_add_(0, cell_ids, torch.ones_like(cell_ids, dtype=pose.dtype))

        sums = torch.zeros([num_cells, pose.shape[-1]], device=self._device, dtype=pose.dtype)
        sums.index_add_(0, cell_ids, pose)

        valid = counts >= self._pose_style_reg_min_count
        means = sums / torch.clamp_min(counts, 1.0).unsqueeze(-1)
        means = means.reshape([num_bins, num_styles, -1])
        valid = valid.reshape([num_bins, num_styles]).type(pose.dtype).unsqueeze(-1)

        # a bin only contributes if at least two styles are represented in it, otherwise there is
        # nothing to compare and the variance is meaningless
        num_valid = valid.sum(dim=1, keepdim=True)
        bin_mask = (num_valid.reshape(-1) >= 2)
        if (not torch.any(bin_mask)):
            return None

        bin_mean = (means * valid).sum(dim=1, keepdim=True) / torch.clamp_min(num_valid, 1.0)
        sq_dev = torch.square(means - bin_mean) * valid
        bin_var = sq_dev.sum(dim=1) / torch.clamp_min(num_valid.squeeze(1), 1.0)

        return torch.mean(bin_var[bin_mask], dim=0)

    def _build_train_data(self):
        self.eval()

        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")
        # the style is fixed for the duration of an episode, and envs are reset only after the
        # step is recorded, so the same style applies to obs and next_obs
        style = self._exp_buffer.get_data("style")

        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs, "style": style}
        next_vals = torch_util.eval_minibatch(self._model.eval_critic, next_critic_inputs, self._critic_eval_batch_size)
        next_vals = next_vals.squeeze(-1).detach()

        succ_val = self._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = self._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, self._discount, self._td_lambda)

        norm_obs = self._obs_norm.normalize(obs)
        critic_inputs = {"obs": norm_obs, "style": style}
        vals = torch_util.eval_minibatch(self._model.eval_critic, critic_inputs, self._critic_eval_batch_size)
        vals = vals.squeeze(-1).detach()
        adv = new_vals - vals

        rand_action_mask = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        rand_action_adv = adv_flat[rand_action_mask]
        adv_mean, adv_std = mp_util.calc_mean_std(rand_action_adv)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)

        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)

        info = {
            "adv_mean": adv_mean,
            "adv_std": adv_std
        }
        return info

    def _compute_critic_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        tar_val = batch["tar_val"]
        pred = self._model.eval_critic(norm_obs, batch["style"])
        pred = pred.squeeze(-1)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        info = {
            "critic_loss": loss
        }
        return info

    def _compute_actor_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_a = self._a_norm.normalize(batch["action"])
        old_a_logp = batch["a_logp"]
        adv = batch["adv"]
        style = batch["style"]
        phase = batch["phase"]
        rand_action_mask = batch["rand_action_mask"]

        # loss should only be computed using samples with random actions
        rand_action_mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[rand_action_mask]
        norm_a = norm_a[rand_action_mask]
        old_a_logp = old_a_logp[rand_action_mask]
        adv = adv[rand_action_mask]
        style = style[rand_action_mask]
        phase = phase[rand_action_mask]

        a_dist = self._model.eval_actor(norm_obs, style)
        a_logp = a_dist.log_prob(norm_a)

        a_ratio = torch.exp(a_logp - old_a_logp)
        actor_loss0 = adv * a_ratio
        actor_loss1 = adv * torch.clamp(a_ratio, 1.0 - self._ppo_clip_ratio, 1.0 + self._ppo_clip_ratio)
        actor_loss = torch.minimum(actor_loss0, actor_loss1)
        actor_loss = -torch.mean(actor_loss)

        clip_frac = (torch.abs(a_ratio - 1.0) > self._ppo_clip_ratio).type(torch.float)
        clip_frac = torch.mean(clip_frac)
        imp_ratio = torch.mean(a_ratio)

        info = {
            "actor_loss": actor_loss,
            "clip_frac": clip_frac.detach(),
            "imp_ratio": imp_ratio.detach()
        }

        if (self._action_bound_weight != 0):
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if (action_bound_loss is not None):
                action_bound_loss = torch.mean(action_bound_loss)
                actor_loss += self._action_bound_weight * action_bound_loss
                info["action_bound_loss"] = action_bound_loss.detach()

        if (self._action_entropy_weight != 0):
            action_entropy = a_dist.entropy()
            action_entropy = torch.mean(action_entropy)
            actor_loss += -self._action_entropy_weight * action_entropy
            info["action_entropy"] = action_entropy.detach()

        if (self._action_reg_weight != 0):
            action_reg_loss = a_dist.param_reg()
            action_reg_loss = torch.mean(action_reg_loss)
            actor_loss += self._action_reg_weight * action_reg_loss
            info["action_reg_loss"] = action_reg_loss.detach()

        if (self._pose_style_reg_weight != 0):
            pose_style_reg_loss = self._compute_pose_style_reg(norm_obs, style, phase)
            if (pose_style_reg_loss is not None):
                actor_loss += self._pose_style_reg_weight * pose_style_reg_loss
                info["pose_style_reg_loss"] = pose_style_reg_loss.detach()

        info["actor_loss"] = actor_loss
        return info

    def _train_iter(self):
        info = super()._train_iter()

        gain_info = self._diag_impedance()
        for k, v in gain_info.items():
            info[k] = v
        return info

    def _diag_impedance(self):
        """Reports how far the impedance is pushed and how style dependent the pd targets are.
        Both are the numbers that decide whether the factorization is working."""
        with torch.no_grad():
            action = self._exp_buffer.get_data_flat("action")
            style = self._exp_buffer.get_data_flat("style")
            obs = self._exp_buffer.get_data_flat("obs")

            pose_size = self._env.get_pose_action_size()
            gain = action[..., pose_size:]

            # fraction of gain actions sitting on the range limit, if this saturates the policy
            # is just asking for the stiffest character it is allowed to have
            bound = self._env.get_action_space().high[pose_size:]
            bound = torch.tensor(bound, device=self._device, dtype=gain.dtype)
            at_bound = (torch.abs(gain) > 0.99 * bound).type(torch.float)

            # cross style spread of the pd targets within a phase bin, against their spread over
            # the whole motion. a small ratio means the styles really do share one equilibrium
            # trajectory, a large one means the style leaked in through the observation
            phase = self._exp_buffer.get_data_flat("phase")
            norm_obs = self._obs_norm.normalize(obs)
            pose = torch_util.eval_minibatch(self._model.eval_pose,
                                             {"obs": norm_obs, "style": style},
                                             self._critic_eval_batch_size)

            # total spread of the targets. the invariance penalty is minimized by a q* that never
            # moves, so this has to be watched alongside the ratio: a ratio that improves while
            # this collapses means the policy bought invariance by standing still
            pose_var_dim = torch.var(pose, dim=0)

            pose_style_var = self._calc_pose_style_var(pose, style, phase)
            if (pose_style_var is None):
                pose_style_ratio = 0.0
            else:
                pose_style_ratio = torch.mean(
                    pose_style_var / torch.clamp_min(pose_var_dim, 1e-6)).item()

            info = {
                "gain_mean_abs": torch.mean(torch.abs(gain)).item(),
                "gain_at_bound": torch.mean(at_bound).item(),
                "pose_style_ratio": pose_style_ratio,
                "pose_var": torch.mean(pose_var_dim).item(),
            }
        return info
