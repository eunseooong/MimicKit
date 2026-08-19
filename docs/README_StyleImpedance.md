# Style as Impedance

Research branch on top of DeepMimic in MimicKit. Two features, layered:

1. **Variable impedance** — the policy outputs PD gains (kp/kd) alongside the joint targets, instead of using the fixed gains from the character file.
2. **Style factorization** — with several phase-aligned motions of the *same content in different styles*, the PD targets are shared across styles and only the impedance carries the style.

Framing borrowed from motor control: the PD target is a **virtual equilibrium trajectory** (Feldman's equilibrium-point hypothesis) and the gains are **impedance** (Hogan, variable impedance control). "Same motor plan, different stiffness" is the claim. Notation used throughout: `q*` = PD targets, `K`/`kp,kd` = impedance, `g` = log-scale gain action.

---

## Status

| | State |
|---|---|
| Variable impedance (engine + env + action space) | implemented, logic unit-tested |
| StyleImp model/agent (shared trunk, style → impedance head only) | implemented, structurally verified |
| Phase-only pose head (`pose_input: "phase"`) | implemented, exact invariance verified |
| Soft invariance penalty (`pose_style_reg_weight`) | implemented, verified on synthetic data |
| `pose_input: "obs_style"` — no structural separation, penalty only (experiment 2) | implemented, structurally verified |
| **Runtime / simulator testing** | **never run** — no Isaac Lab installed in the dev environment |
| Offline feasibility test (rank-1, see below) | **not done** — recommended before trusting any training result |

Everything below was verified by reading source and by unit tests on the pure logic. **Nothing has been through the simulator.** Numbers marked *(measured)* come from real forward/backward passes on CPU with fake envs; numbers marked *(derived)* come from the configs.

---

## Architecture

### Layer 1 — engine: who computes the torque

`τ = kp · (q* − q) − kd · q̇`, clamped to `±effort_limit` (from `<motor gear=...>` in the MJCF). Evaluated at `sim_freq`; the policy updates targets and gains at `control_freq`.

| Engine | `control_mode` | Gains live in | Runtime gain writes |
|---|---|---|---|
| Isaac Lab | `pos` | PhysX (`ImplicitActuator`) | `write_joint_stiffness_to_sim` — **full buffer copied to CPU every call** |
| Isaac Lab | `pd_explicit` | GPU torch buffers (`IdealPDActuator`) | `actuator.stiffness[:] = ...` — plain GPU write |
| Isaac Gym | `pd_explicit` | `_kp_raw` / `_kd_raw` tensors | in-place view write, PD computed in `_calc_pd_explicit_torque` |
| Isaac Gym | `pos` | PhysX DOF properties | **not possible** — per-actor CPU API only. `supports_variable_gains()` returns False and startup asserts |

New engine API (`mimickit/engines/engine.py`):

```python
def set_gains(self, obj_id, kp, kd)     # kp, kd: [num_envs, num_dofs], common dof order
def supports_variable_gains(self)       # False in the base class
```

### Layer 2 — env: action space and gain decode

`mimickit/envs/char_env.py`

```
action = [ q* (69) | g (gain dims) ]

kp = kp_nom · exp(g)        g bounded to ±ln(impedance_range)
kd = kd_nom · exp(g)        then kd = max(kd, kp · Δt_sim)
```

- `_build_impedance_params()` — builds the dof→gain-group map, caches nominal gains from the character file. Called from `_build_action_space()` because that is the first hook that runs *after* `initialize_sim()`.
- `_append_impedance_bounds()` — extends the Box with the gain dims.
- `_calc_impedance()` — exp, broadcast, kd floor.
- `_apply_action()` — clamps, splits at `dof_size`, calls `set_gains` then `set_cmd`.
- `_print_impedance()` — per-joint gain table in TEST mode, every `impedance_print_int` steps.

New public getters: `get_pose_action_size()` (69), `get_impedance_action_size()`.

`mimickit/envs/deepmimic_env.py`

- `_build_styles()` — maps `motion_id` → style index. Called after `super().__init__()`, so `_motion_lib` and `_motion_ids` both exist.
- `get_num_styles()`, `get_style_ids()`, `get_style_onehot()`
- `get_motion_phase()` — scalar phase in [0,1], recorded per step for the invariance penalty.
- `get_phase_obs_range()` — index range of the phase encoding inside the obs. Asserts `enable_phase_obs and not enable_tar_obs`, because that is what puts the phase at the tail.

### Layer 3 — learning: the two-lane actor

`mimickit/learning/style_imp_model.py`, `style_imp_agent.py`, registered as `agent_name: "StyleImp"`.

```
                      ┌──► pose net ──────────────────► q*  (69)
phase (25 dims) ──────┘

obs ──► trunk (1024) ──┐
                       ├──► imp net (256) ────────────► g   (23)
style one-hot ─────────┘

concat mean + logstd  ──►  one DistributionGaussianDiag(92)  ──► PPO, unmodified

critic:  [ obs ; style ] ──► V
```

The single combined distribution is deliberate: `log_prob`, PPO clipping, and `_compute_action_bound_loss` all work without modification.

`pose_input` decides what the pose head may see, and it is the whole experiment:

| `pose_input` | Actor shape | Pose head input | Guarantee |
|---|---|---|---|
| `"phase"` | pose head + imp head, style → imp head only | phase encoding only | **exact** — `Δq* = 0`, `d(q*)/d(trunk) = 0` *(measured)* |
| `"obs"` | pose head + imp head, style → imp head only | full obs via the shared trunk | **none** structurally; style leaks via the state |
| `"obs_style"` | **one trunk over `[obs ; style]`, one head over all 92 dims** | full obs **and** the style label | **none** at all; the penalty is the only constraint |

`"obs_style"` is the stock DeepMimic actor with the style appended to the observation — no routing, no extra heads. `pose_net` / `imp_net` are unused. *(measured: `d(q*)/d(style) = 1.63e+01`, i.e. the style reaches the targets directly, as intended for that experiment.)*

---

## The central pitfall (read this before changing `pose_input`)

Cutting the style *label* out of the pose path is **not** enough to make `q*` style-invariant.

The observation contains the character state, and the state differs between styles. A pose head that reads the observation can infer the style from the state and emit a style-specific target. Routing the label away guarantees only that `q*` is one *function* of the observation — **not** that it is one *trajectory*.

Worse, that degenerate solution is the **easier** one for the optimizer: 69 target dims act on the tracking reward directly, while 23 gain dims can only rescale a deviation whose direction the gravity/inertia load already fixed. Expect "identical impedance, style-specific `q*`" as the default outcome unless it is structurally prevented.

Measured with the same phase encoding and deliberately different character states:

```
pose_input = "obs"     max |Δq*| across styles = 4.82e+01
pose_input = "phase"   max |Δq*| across styles = 0.00e+00
```

**A shared trunk and exact invariance are mutually exclusive**, because the trunk reads the observation and the observation carries the style. Only input restriction buys an exact guarantee.

If the phase-only head cannot balance (plausible for dynamic motion — gains scale a restoring force but never reverse it), fall back to `pose_input: "obs"` with `pose_style_reg_weight > 0` and report the result as *approximate* factorization.

---

## Config reference

### Env (`data/envs/*.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `impedance_mode` | `"none"` | `none` / `coupled` (one scale per group, nominal kd/kp ratio kept) / `decoupled` (independent kp and kd scale) |
| `impedance_range` | `4.0` | Gains span `[1/R, R]` × the character file values. Action bounds are `±ln(R)` |
| `impedance_per_dof` | `false` | `false` = one gain per joint (23), `true` = one per dof (69) |
| `multi_dof_pd` | `false` | Opt out of the `pd_explicit` 1-dof-joint assert. Required for SMPL on `pd_explicit` |
| `impedance_print_int` | `30` | Control steps between per-joint gain printouts in TEST mode, 0 disables |
| `style_ids` | `None` | Motion index → style index. Omit for one style per motion. Must cover `0..N-1` with no gaps |

`impedance_mode: "none"` is a complete no-op, so every pre-existing config behaves exactly as before.

### Agent (`data/agents/*.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `model.pose_input` | `"phase"` | See the table above |
| `model.pose_net` | `actor_net` | Pose head net, only used when `pose_input: "phase"` |
| `model.imp_net` | `actor_net` | Impedance head net, unused when `pose_input: "obs_style"` |
| `pose_style_reg_weight` | `0.0` | Soft invariance penalty. Pointless with `"phase"`, the whole mechanism with `"obs_style"` |
| `pose_style_reg_bins` | `20` | Phase bins for that penalty |
| `pose_style_reg_min_count` | `2` | Minimum samples for a `(bin, style)` cell to count |
| `pose_style_reg_normalize` | `True` | Divide the per-dim cross-style variance by that dim's total variance |

### Action layout *(derived, SMPL)*

| `impedance_mode` | `per_dof` | Gain dims | Action dims |
|---|---|---|---|
| `coupled` | `false` | 23 | **92** |
| `decoupled` | `false` | 46 | 115 |
| `coupled` | `true` | 69 | 138 |
| `decoupled` | `true` | 138 | 207 |

---

## Design decisions, and why

**Gains are log-scale multipliers of the character file values, with symmetric bounds.**
`_a_norm` maps the Box linearly to `[-1, 1]`, so a symmetric bound puts the range centre at `g = 0` → exactly 1× nominal. Combined with `actor_init_output_scale: 0.01`, a freshly initialised policy starts at the original stiffness rather than somewhere arbitrary — the same initial condition as a normal `pos`-mode run. A linear parameterisation would only land on nominal by coincidence. *(verified: zero action reproduces `kp_nom`/`kd_nom` exactly)*

**Gains are per joint, not per dof, by default.**
`(kp, q*)` produce the same torque along a manifold, so per-dof gains make the pair maximally ill-conditioned. Per-joint scalars keep only the physically meaningful "stiffen this joint" degree of freedom.

**`kd = kp/10` holds for every joint in `smpl.xml`** *(verified across all 69 joint entries)*, so `coupled` mode preserving that ratio is not an arbitrary constraint — and it is the main reason `coupled` is safer than `decoupled` on explicit-PD engines, where a policy that drives kd down while kp up will diverge.

**Per-axis gains would change the deviation *direction*, not just its magnitude.**
With one gain per joint, `Δ = −(1/K)(h_x, h_y, h_z)` — a fixed direction, variable length. With per-axis gains, `Δ = −(h_x/K_x, h_y/K_y, h_z/K_z)` — the direction rotates, but only inside the octant cone containing the load (K > 0 cannot flip a sign) and only within `[1/R, R]` per component. This is anisotropic joint impedance; the human "stiffness ellipse" literature is the precedent. A true ellipse needs a full 3×3 stiffness matrix per joint, which per-dof gains cannot express — that would require a custom torque path.

**DeepMimic rather than AMP for the multi-style work.**
Each env is rewarded for tracking the motion it sampled, so per-style pressure is already in the reward. A single AMP discriminator learns the union of all styles and applies **no** per-style pressure at all — that would need a style-conditioned discriminator first.

**The critic sees the style; only the pose head is kept blind.**
Value depends on which style is being tracked. The disentanglement claim is about `q*`, not about `V`.

**`enable_tar_obs: False`, `enable_phase_obs: True`.**
Target observations carry the style-specific reference pose — a second leak channel — and would displace the phase encoding from the tail of the obs, which `get_phase_obs_range()` depends on.

**`num_phase_encoding: 12`.**
With `pose_input: "phase"`, `q*` can only express what the phase encoding resolves. Four harmonics cannot represent a sharp hip-hop move.

---

## Known traps

**`obs_version` is ignored by DeepMimicEnv.** `CharEnv` dispatches on it (`char_env.py:412`), but `DeepMimicEnv._compute_obs` overrides the whole path and calls `char_env.compute_char_obs` — the v1 function — directly at `deepmimic_env.py:747`. Setting `obs_version: "v2"` in a DeepMimic env config **silently does nothing**. `style_imp_smpl_env.yaml` currently has `v2` set; it is running v1.

**`data/engines/isaac_lab_pd_engine.yaml` is named and commented for `pd_explicit` but currently sets `control_mode: "pos"`.** Both work; the file just no longer says what it does.

**PhysX cannot do sparse gain writes.** `set_dof_stiffnesses` docstring: *"The sparse setting of subset of DOFs within an articulation is not supported yet."* Passing `env_ids`/`joint_ids` to `write_joint_stiffness_to_sim` still sends the full `[num_envs, max_dofs]` buffer, and Isaac Lab `.cpu()`s it (`articulation.py:638`, `:667`). At 4096 envs that is ~2.3 MB and two GPU→CPU syncs per control step. The bandwidth is trivial; the pipeline stall is not. This is not fixable by patching Isaac Lab. Whether the underlying PhysX call accepts a CUDA tensor was **not** verified — the type union allows it, every docstring example uses `device="cpu"`.

**Gain writes are not reverted, though.** `ImplicitActuator.compute()` is a no-op for control (`actuator_pd.py:117-140`) and gains are written to sim only once at init (`articulation.py:1734-1735`). Isaac Lab's own domain-randomisation event uses the same write pattern (`envs/mdp/events.py:625`) — but at reset intervals, not per control step.

**`actuator.computed_effort` / `applied_effort` go stale.** `write_joint_stiffness_to_sim` deliberately does not update the actuator model's buffers (upstream issue #128). Convenient for us — `get_obj_pd_gains()` keeps returning nominal, which is what the env caches — but any future torque penalty or energy reward that reads applied effort will be **silently wrong**. Note also `_enable_dof_force_sensors()` is hardcoded `False` in the Isaac Gym engine.

**Two pressures act on the gains, and both are blunt.** `action_bound_weight: 10.0` only bites outside `[-1, 1]` in normalized space, so it does not stop the policy from parking just inside the limit. `action_reg_weight` *is* wired in `ppo_agent` (`_compute_actor_loss`, via `DistributionGaussianDiag.param_reg` = `sum(mean²)`) and does pull the gains toward nominal, since `g = 0` is nominal — but it applies to **all 92 action dims**, so it simultaneously pulls `q*` toward the middle of its range and flattens the motion. A gain-only regulariser would have to slice the action before squaring. Watch `gain_at_bound`.

**`pose_style_ratio` must be phase-binned.** Comparing per-style means over the whole motion averages the leak away and reports a healthy number for a broken policy. `_calc_pose_style_var()` bins by phase; both the diagnostic and the penalty share it.

---

## The soft constraint (experiment 2)

Penalty on the pd targets, added to the actor loss. `_calc_pose_style_var()` bins the batch by phase and, per output dim, takes the variance across the per-`(bin, style)` mean targets, averaged over bins that hold at least two styles. Returns `None` when no bin qualifies, and the penalty is then skipped rather than contributing a bogus number.

```
loss = mean_d [ cross_style_var_d / total_var_d ]        (normalize: True)
     = sum_d    cross_style_var_d                        (normalize: False)
```

**What normalization buys** *(measured on synthetic targets)*: wide-range joints stop dominating, and the loss lands on the same `[0, ~1]` scale as `pose_style_ratio`, so the weight carries over between motions. Style-dependent targets score `1.03`, phase-only targets `1.5e-03` — a ~670× separation.

**What it does not buy — verified, this was initially claimed and is false.** Collapsing `q*` to a constant zeroes the numerator *and* the denominator, so the ratio goes to 0 exactly like the raw variance does: `collapse = 0.0` vs `good = 1.5e-03`. **Both forms reward a policy that stops moving.** Only the tracking reward pushes back. `pose_var` is logged to detect it.

**The penalty needs many styles per minibatch.** It compares styles *within* a phase bin, so a minibatch must contain several styles at the same phase. With few envs, or with `num_bins` set high relative to the batch, most cells fall below `min_count` and the penalty silently does nothing. Check that `pose_style_reg_loss` actually appears in the logs.

---

## Diagnostics

Logged every iteration by `StyleImpAgent._diag_impedance()`.

| Metric | Reads | Bad sign |
|---|---|---|
| `pose_style_ratio` | cross-style variance of `q*` inside a phase bin ÷ its total variance, per dim | rising — style is leaking into the targets |
| `pose_var` | mean per-dim variance of `q*` | falling toward 0 — the policy bought invariance by standing still |
| `gain_at_bound` | fraction of gain actions within 1% of `±ln R` | → 1, the policy just wants the stiffest character allowed |
| `gain_mean_abs` | mean magnitude of the log multiplier | → 0, the impedance is carrying nothing |

With `pose_input: "phase"`, `pose_style_ratio` is 0 by construction — it is only informative in `"obs"` and `"obs_style"` mode. **Read it together with `pose_var`**: a ratio that improves while `pose_var` collapses is not a success.

---

## Measuring a trained model: `tools/eval_style_q.py`

```bash
python tools/eval_style_q.py --arg_file args/style_imp_soft_smpl_args.txt \
    --model_file output/style_imp_soft/model.pt --num_envs 512 --steps 400 \
    --num_bins 20 --out_file output/style_imp_soft/style_q.npz
```

Rolls the policy out over every style, bins the pd targets by phase, and reports how far apart the styles' targets sit. Three readings, because "how far apart" is ambiguous:

| Reading | Answers |
|---|---|
| **on-policy** | do the styles actually share one equilibrium trajectory? This is the claim, and it is what the training penalty targets |
| **counterfactual** | same observation, style label swapped. Isolates the *direct* label channel from the *state* channel. In `"obs_style"` both are open and only this split says which does the work; in `"phase"`/`"obs"` it is 0 by construction |
| **vs reference** | on-policy spread ÷ the spread of the reference motions at the same phase. The interpretable number: ratio ≈ 1 means the targets differ as much as the motions do and the factorization achieved nothing; ratio ≪ 1 means the impedance is carrying the difference |

Also prints a per-style-pair distance matrix (one outlier style vs a uniform spread), a per-phase-bin curve (*where* in the motion they diverge), and the worst joints (*which* joints diverge).

**The metric has a noise floor, and the tool measures it for you.** A phase bin averages over a range of phases, so a fast trajectory leaves a residual even when the styles agree exactly. `within_style_floor()` splits each style's *own* samples in half and measures the spread between the halves — same binning, same sample count, zero true style effect. The report divides by it: **a signal/floor near 1 means the styles are indistinguishable in `q*`.** *(measured on synthetic data: identical styles → 1.34, a 0.01 rad offset → 4.19, a 0.30 rad offset → 114.7, so the floor resolves well below anything meaningful.)* More bins lowers the floor but empties cells.

The math lives in `mimickit/util/style_metrics.py`, free of any simulator import so it can be unit tested without Isaac.

---

## Experiment plan

**E0 — offline feasibility, not yet done, do this first.**
Purely kinematic, no RL, no simulator. Under the gain-only hypothesis `q_s(φ) ≈ q*(φ) − h(φ)/K_s`, so per joint `j` the matrix `M_j[s, t] = q_s(φ_t)[j] − q̄(φ_t)[j]` must be **rank 1**:

```python
delta = q - q.mean(axis=0, keepdims=True)          # q: [styles, frames, 69]
for j in range(69):
    s = np.linalg.svd(delta[:, :, j], compute_uv=False)
    ratio[j] = s[0]**2 / (s**2).sum()               # want >= 0.8
```

Three outcomes: rank-1 fails → gain modulation cannot express the style at any granularity, stop. Rank-1 holds and the per-style coefficient is consistent across a joint's three axes → per-joint gains suffice. Rank-1 holds but the coefficients differ per axis → `impedance_per_dof: true` is needed, and the test says for which joints. Needs ≥5 styles to be meaningful, and also yields initial values for `q*` and `K_s`.

**E1** — `pd_explicit` wiring check: single motion, gains fixed at nominal, confirm parity with a normal `pos` run. Everything downstream sits on this.

**E2** — single motion, learn `q*` and gains jointly, no disentanglement. Does variable impedance train at all, and does `gain_at_bound` stay off 1? Start with `impedance_range: 1.5`, then widen to 4.

**E3 — structural.** Multi-style with `pose_input: "phase"`. Invariance is exact and free; the risk is that a pose path with no state feedback cannot balance.

**E3b — soft, no structure.** `pose_input: "obs_style"`: stock DeepMimic actor with the style appended, and the penalty as the *only* constraint. Strictly weaker prior than E3 and a strictly harder test — the style label reaches `q*` directly here, so nothing but the penalty stops style-specific targets. Sweep `pose_style_reg_weight` (0 → 0.1 → 1 → 10) and plot reward against `pose_style_ratio`; the interesting output is that trade-off curve, not a single run. **Keep the weight-0 run** — showing that the degenerate solution actually happens is itself a result, and it is the only baseline that makes the other points mean anything.

Between the two there is a third rung, `pose_input: "obs"` + penalty: the style *label* is routed away but the state still leaks. Useful if E3b's curve is bad and E3 will not stand, since it stacks the partial structural prior with the penalty.

**E4 — the real evaluation.** Swap style A's rollout gains for style B's: does the style swap? Interpolate the style one-hot: are the intermediates plausible? *(the model supports interpolation — verified)* If swapping gains does not swap style, the factorization failed regardless of the reward curve.

For the frozen-`q*` transfer variant (train on A, then re-learn only the impedance for B), the three-point comparison is the deliverable: `(experiment − zero-shot floor) / (full fine-tune ceiling − floor)` = fraction of the A→B gap closed by impedance alone. A single reward number means nothing.

---

## Assumptions this all rests on

**Phase alignment.** The shared trajectory, the phase-binned penalty, and the diagnostic all assume equal phase = same point of the motion across styles. If takes differ in length or loop mode, `calc_motion_phase` drifts and the whole construction measures noise. **Verify before the first run.**

**The motion set really is one content in several styles.** `data/datasets/dataset_smpl_happyfeet_styles.yaml` currently lists 29 takes from `2025_01_MOVIN_HIPHOP2/01-30-happyfeet-1miss`. If those are repeat recordings rather than distinct styles there is nothing to factorize. Trim to 3–5 styles for a first run; the one-hot grows with the count and `pose_style_ratio` gets harder to read.

**Gain range.** If a style is unreachable within `[1/R, R]` the experiment fails for a trivial reason. Sweep R and report which value was needed. Large R mostly saturates the effort limit rather than stiffening further.

**Impedance scales, it does not redirect.** With per-joint gains the deviation from `q*` is fixed in direction by the load. Styles differing in compliance, sharpness, or effort are expressible; styles whose limbs go elsewhere at the same phase are not. E0 is the cheap way to find out which kind you have.

---

## Files

**New**

```
mimickit/learning/style_imp_model.py     two-lane actor
mimickit/learning/style_imp_agent.py     PPO subclass, style plumbing, diagnostics, penalty
data/agents/style_imp_smpl_agent.yaml            experiment 1: pose_input "phase"
data/agents/style_imp_soft_smpl_agent.yaml       experiment 2: pose_input "obs_style" + penalty
data/envs/style_imp_smpl_env.yaml                shared by both
data/envs/{amp,deepmimic}_smpl_imp_env.yaml     single-motion impedance configs
data/engines/isaac_lab_pd_engine.yaml
data/datasets/dataset_smpl_happyfeet_styles.yaml
args/style_imp_smpl_args.txt
args/style_imp_soft_smpl_args.txt
args/{amp_smpl_imp,deepmimic_smpl_imp_ppo}_args.txt
```

**Modified**

```
mimickit/engines/engine.py               set_gains / supports_variable_gains base
mimickit/engines/isaac_lab_engine.py     both gain-write paths
mimickit/engines/isaac_gym_engine.py     pd_explicit gain write
mimickit/envs/char_env.py                ImpedanceMode, action space, gain decode, printout
mimickit/envs/deepmimic_env.py           style table, phase getters
mimickit/learning/agent_builder.py       "StyleImp" registration
```

## Running

```bash
python mimickit/run.py --arg_file args/style_imp_smpl_args.txt           # exp 1: structural (phase-only q*)
python mimickit/run.py --arg_file args/style_imp_soft_smpl_args.txt      # exp 2: penalty only
python mimickit/run.py --arg_file args/deepmimic_smpl_imp_ppo_args.txt  # single motion, impedance only
```
