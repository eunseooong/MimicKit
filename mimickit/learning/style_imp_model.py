import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.distribution_gaussian_diag as distribution_gaussian_diag
import learning.nets.net_builder as net_builder
import learning.ppo_model as ppo_model
import util.torch_util as torch_util

class StyleImpModel(ppo_model.PPOModel):
    """Actor that splits the action into pd targets and impedance, with the style wired only into
    the impedance head.

    pose_input controls what the pose head is allowed to see, and it is the whole ballgame:

    "phase" -- the pose head sees only the phase encoding. Since phase-aligned styles share the
        phase, the pd targets are then *literally identical* across styles at the same point in
        the motion. This is an exact guarantee that needs no measurement, and it is the only way
        to get one. The cost is that the pose head has no state feedback and cannot share the
        trunk, so all closed-loop correction has to come from the impedance.

    "obs" -- the pose head sees the full observation through a trunk that the impedance head also
        uses. Cutting the style label out of this path guarantees only that the pd targets are one
        *function* of the observation, NOT that they are one *trajectory*: the character state
        differs between styles, so the head can read the style off the state and emit a
        style-specific target. That degenerate solution is not a corner case, it is the easier
        one for the optimizer, because 69 target dims act on the reward far more directly than 23
        gain dims that can only rescale a load-determined deviation. Use this only together with
        pose_style_reg_weight, and watch pose_style_ratio.
    """

    def __init__(self, config, env):
        self._pose_size = env.get_pose_action_size()
        self._imp_size = env.get_impedance_action_size()
        self._num_styles = env.get_num_styles()
        self._pose_input = config.get("pose_input", "phase")

        assert(self._imp_size > 0), "StyleImp requires an env with impedance_mode enabled"
        assert(self._pose_size + self._imp_size == np.prod(env.get_action_space().shape))
        assert(self._pose_input in ["phase", "obs"]), \
            "Unsupported pose_input: {}".format(self._pose_input)

        if (self._pose_input == "phase"):
            self._phase_beg, self._phase_end = env.get_phase_obs_range()

        super().__init__(config, env)
        return

    def eval_actor(self, obs, style):
        h = self._actor_layers(obs)

        pose_dist = self._pose_dist(self._eval_pose_feat(obs, h))

        imp_h = self._imp_layers(torch.cat([h, style], dim=-1))
        imp_dist = self._imp_dist(imp_h)

        # the agent works with one distribution over the full action, the split is internal
        mean = torch.cat([pose_dist.mean, imp_dist.mean], dim=-1)
        logstd = torch.cat([pose_dist.logstd, imp_dist.logstd], dim=-1)
        a_dist = distribution_gaussian_diag.DistributionGaussianDiag(mean=mean, logstd=logstd)
        return a_dist

    def eval_pose(self, obs):
        # pd targets on their own, for the invariance penalty and the diagnostics
        if (self._pose_input == "phase"):
            feat = self._eval_pose_feat(obs, None)
        else:
            feat = self._eval_pose_feat(obs, self._actor_layers(obs))
        return self._pose_dist(feat).mode

    def _eval_pose_feat(self, obs, trunk_h):
        if (self._pose_input == "phase"):
            phase_obs = obs[..., self._phase_beg:self._phase_end]
            return self._pose_layers(phase_obs)
        return trunk_h

    def eval_critic(self, obs, style):
        h = self._critic_layers(torch.cat([obs, style], dim=-1))
        val = self._critic_out(h)
        return val

    def get_actor_params(self):
        params = list(self._actor_layers.parameters()) \
                 + list(self._pose_dist.parameters()) \
                 + list(self._imp_layers.parameters()) \
                 + list(self._imp_dist.parameters())
        if (self._pose_input == "phase"):
            params += list(self._pose_layers.parameters())
        return params

    def _build_actor(self, config, env):
        net_name = config["actor_net"]
        input_dict = self._build_actor_input_dict(env)
        self._actor_layers, _ = net_builder.build_net(net_name, input_dict,
                                                      activation=self._activation)

        trunk_size = torch_util.calc_layers_out_size(self._actor_layers)

        if (self._pose_input == "phase"):
            # the pose head gets its own net, it must not touch the trunk because the trunk reads
            # the observation and the observation carries the style
            phase_size = self._phase_end - self._phase_beg
            pose_net_name = config.get("pose_net", net_name)
            pose_input_dict = {"phase": spaces.Box(low=-1.0, high=1.0, shape=[phase_size])}
            self._pose_layers, _ = net_builder.build_net(pose_net_name, pose_input_dict,
                                                         activation=self._activation)
            pose_feat_size = torch_util.calc_layers_out_size(self._pose_layers)
        else:
            pose_feat_size = trunk_size

        self._pose_dist = self._build_head_dist(config, pose_feat_size, self._pose_size)

        imp_net_name = config.get("imp_net", net_name)
        imp_input_dict = {
            "h": spaces.Box(low=-np.inf, high=np.inf, shape=[trunk_size]),
            "style": self._build_style_space(),
        }
        self._imp_layers, _ = net_builder.build_net(imp_net_name, imp_input_dict,
                                                    activation=self._activation)

        imp_size = torch_util.calc_layers_out_size(self._imp_layers)
        self._imp_dist = self._build_head_dist(config, imp_size, self._imp_size)
        return

    def _build_head_dist(self, config, in_size, out_size):
        init_output_scale = config["actor_init_output_scale"]
        std_type = distribution_gaussian_diag.StdType[config["actor_std_type"]]
        init_std = config["action_std"]
        a_dist = distribution_gaussian_diag.DistributionGaussianDiagBuilder(
            in_size, out_size, std_type=std_type, init_std=init_std,
            init_output_scale=init_output_scale)
        return a_dist

    def _build_critic_input_dict(self, env):
        # the value of a state depends on which style is being tracked, so the critic sees it
        input_dict = {"obs": env.get_obs_space(),
                      "style": self._build_style_space()}
        return input_dict

    def _build_style_space(self):
        style_space = spaces.Box(low=0.0, high=1.0, shape=[self._num_styles])
        return style_space
