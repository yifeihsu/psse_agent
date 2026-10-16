# Review of the RL implementation and of the case for switching methods (2026-10-16)

Branch `codex/classifier-triage-20261004`. Answers the question of 2026-10-16:
the RL arms show no significant gain; is the implementation correct, and would
another method, such as GAGPO (Generalized Advantage Grouped Policy
Optimization, arXiv 2605.13217), do better? Read against the code at `1f70822`
(`psse_env/rl/`, the sampler in `psse_env/dagger/research_policy_factory.py`,
the cell under `research/hpc/rl_pilot_20261014/`) and the run records in
`docs/rl_experiments_20261014.md`, sections 3 and 5 (the pilot on the standard
draw, the two hard-draw arms after two iterations, and the stratified,
KL-anchored rerun of the screen arm). The cluster outputs themselves were not
reachable from here; everything quantitative about the runs is taken from that
record. Section 6 lists what this review added to the code.

## 1. Verdict in one paragraph

The implementation is correct as an on-policy GRPO/RLOO loop: the sampler draws
from the policy's own distribution at the configured temperature with no
nucleus or top-k cut, each sample has its own seed, the environment is
deterministic given the root so the four samples of a group differ only through
the policy's draws, the training rows hold the exact sampled tokens including
the stop token behind a prompt re-rendered by the same renderer, the loss has
the right sign and the DAPO normalization, the KL estimator and the reference
adapter are right, and the checkpoint resume reproduces an uninterrupted update.
Two stated assumptions are not quite kept: the "importance ratio one" claim
holds for the first optimizer step of a batch only, and the configs take two,
and the training forward pass has never been checked against the decoding pass
numerically. Neither explains the results. What does is upstream of the
estimator: on 69 to 90% of root groups the four samples score the same, so
there is no gradient; in the groups that do differ, the difference is mostly
one sample paying a cost or making a mistake rather than a success flip, so the
gradient sharpens the greedy policy or, as the screen arm's second update
showed, follows the cost terms off a near-tie decision; the group-standardized
advantage gives a one-call difference the same weight as a success flip; the
token-level normalization let one 40-step sample outweigh the rest of a batch;
and each run so far amounts to four AdamW steps at learning rate 1e-5 against
a DAgger round's about 280 steps at 3e-5. GAGPO leaves the first and the last
of these untouched. It does fit this environment unusually well: with
deterministic observations and a terminal reward its step-level advantage puts
a deviating sample's whole credit on the one step where it left the others,
which is exactly what would have kept the screen arm's 1,014-token looping
sample from steering a step. So it is a sound next change to the estimator,
after the signal, the scale and the budget, not instead of them.

## 2. What was checked

| Component | Where | Finding |
| --- | --- | --- |
| Sampling distribution | `research_policy_factory.py`, `_generation_kwargs` | `do_sample=True`, temperature as configured, `top_p` 1.0 and `top_k` 0 passed explicitly, so Gemma 4's generation defaults (top-k 64, top-p 0.95) never apply. The config refuses any cut. Correct. |
| Per-sample seeding | `rollout.py`, `sample_seed`; `research_gemma_policy_factory` | Every sample gets a fresh policy object with `seed = sha256(base, root, index)`; only the weights are cached. Each `generate` call reseeds with `seed + call_index`. Correct. |
| Environment determinism | `scripts/run_dagger_research.py`, `research_diagnostic_environment_factory` (`del seed, rng`) | The evaluator passes a per-sample seed, the factory discards it; the scenario carries its measurements. A group's samples therefore share the state distribution exactly, which is what a group-relative baseline needs. Correct. |
| Recorded tokens and probabilities | `generate_text`, `_record_generation`, `sampled_token_probabilities` | The completion is the generated sequence up to and including the stop token (`decode_generated_response` trims only trailing pads); the probability of each token is the softmax of `generate`'s processed scores, which is the sampling distribution. Correct. |
| Training rows | `rows.py` | Prompt re-rendered from the stored observation with the policy's renderer and checked against the recorded token count; completion = the recorded tokens; assistant-only mask; side inputs zero. Correct. A content hash would be stricter than the length check, but the renderer is deterministic over a sorted JSON payload. |
| Loss | `update.py`, `completion_logprobs`, `policy_gradient_update` | `-A * log pi` summed over completion tokens, divided by the optimizer step's completion tokens (DAPO); log-probabilities at the sampling temperature from the last `n + 1` logits; sign verified by test. Correct. |
| Advantages | `advantages.py` | GRPO: reward minus group mean over the group's population standard deviation (`scale: group`), or unscaled; RLOO: leave-one-out mean. Zero-variance groups dropped. Correct. |
| KL and reference | `peft_reference_logprobs` | k3 estimator of KL(pi_theta || pi_ref) on the sampled tokens; the reference is the starting adapter as a second PEFT adapter, trainable flags restored. Correct; `beta_kl` 0 in the first arms, 0.04 in the rerun. |
| Model in the update | `psse_env/sft/research_cli.py`, `attach_research_adapter` | 4-bit base, `prepare_model_for_kbit_training`, adapter loaded trainable from its own config. LoRA dropout is 0 in BC0 and the rounds (`stage_train.sbatch`), so `model.train()` adds no noise. Correct. |
| Root selection | `train.py`, `iteration_roots`; `sampling.py` (1f70822) | Disjoint slices of one seeded permutation, or fixed counts per hard-draw stratum. Both deterministic; the strata come from the audit and never reach the policy. Correct. |
| Checkpointing | `UpdateCheckpoint`, tests | An update interrupted mid-step resumes to the same parameters (tested on a bigram model, with and without the row bound). Correct. |
| Evaluation | `train.py`, `phase_evaluate` | Greedy, the same seed and contract as every cell, stored per root, skipped when the adapter did not change. Correct. |
| Tests | `python -m pytest -q psse_env/rl research/rl` | 56 passed, 4 skipped at `592cac0` under the pinned versions (torch 2.10.0 CPU, numpy 2.3.5, scipy 1.16.3, PYPOWER 5.1.19, fastmcp 2.12.4); 76 passed, 4 skipped with this review's tests and the stratified-sampling tests of `1f70822`. The skips need transformers and PEFT or the local suite. |

## 3. Defects and loose ends, in order of consequence

None of these changes the conclusion of section 4, but each is worth fixing or
knowing, and the first two acted in the runs.

1. **The token normalization lets one long sample set a step.** Under the
   DAPO normalization a trajectory contributes one row per step and every
   row's tokens, so a 40-step looping sample weighs as much as seven 6-step
   ones. The screen arm's iteration 1 is the case: one such sample
   (advantage -1.73, 1,014 tokens) produced a 22.8 row norm and most of the
   45.8 step norm, and the step broke the harmonic decision (12 to 5 of 16
   roots). The row bound at 25 could not catch it: the sample's weight is
   spread over 40 rows, each under the bound. The unbiased REINFORCE gradient
   does sum over every step of a trajectory, so this is a variance problem,
   not a bug; but with four samples per root and ten to twenty informative
   groups a batch cannot afford it. Two remedies: `loss_normalization:
   "trajectory"` (added, section 6), under which every trajectory of a step
   weighs the same whatever its length; or a step-level advantage that
   credits a lone deviating sample at the step where it deviated only
   (section 5). Lowering the row bound does not help.
2. **The group-standardized advantage equalizes noise and signal.** With
   rewards {x, x, x, x - d} the GRPO advantages are +0.577 (three times) and
   -1.732 for any d. One extra call costs d = 0.04 (call cost 0.02 plus regret
   0.02); an unneeded acquisition 0.14; a success flip about 1. All three
   groups move the parameters by the same amount. In the screen arm's
   iteration 1, 18 of the 20 informative groups were measurement-type roots
   whose spread came from the cost terms (a sample fetching the three-phase or
   harmonic context it did not need), and the update generalized "do not
   fetch the harmonic context" to the harmonic roots, where it is the
   diagnosis. In reward units those 18 groups would have carried advantages
   of about 0.1 against the harmonic groups' about 1; standardized, they
   carried the same. This is the bias Dr. GRPO (arXiv 2503.20783) removes by
   dropping the standard deviation; `advantage_scale: "none"` already exists
   in the code and keeps the advantages in reward units. RLOO with `scale:
   none` is the same direction scaled by K/(K - 1).
3. **Two optimizer steps per batch without a ratio.** `update.py` documents
   "one on-policy gradient pass (ratio one), so no clipping is needed"; every
   config sets `optimizer_steps: 2`. The second step evaluates the plain
   `A * log pi_theta` on rows sampled from the parameters before the first
   step, so it is a small off-policy gradient with no correction. At learning
   rate 1e-5 the drift is tiny, but the second step is where the large norms
   appeared (5,255 in the pilot, 65.7 and 45.8 on the hard draw). Either set
   `optimizer_steps: 1`, or use the recorded sampling probabilities as
   `pi_old` with a PPO clip; the latter is now `update.clip_ratio` (section
   6) and is what every GRPO descendant does when a batch is used for more
   than one step.
4. **The decoding pass and the training pass were never compared.** The
   sampler runs a cached, incremental bf16 decode with the adapter in
   inference mode; the update runs a full-sequence forward through
   `prepare_model_for_kbit_training` (norms and embeddings in float32) with
   gradient checkpointing. The two agree in exact arithmetic and differ
   numerically; the size of the gap is unknown, and on rows whose tokens have
   probability near one the gradient is carried by the few tokens where such a
   gap would matter most. The update now reports it per optimizer step
   (`on_policy_check` in `update_report.json`): at the first step of a batch
   the gap is purely numerical, from the second step on it includes the drift
   of item 3. Expect a mean absolute log-ratio well below 0.05 at step one;
   anything larger points at a tokenizer, template or dtype mismatch between
   the two paths. The KL smoke test of the rerun checks the same thing from
   the other side (a first-step KL of zero against the reference adapter).
5. **Unpaired trajectories stay in the baseline and leave the loss.** A
   trajectory whose generation records do not pair with the trace is excluded
   from the rows but its reward still enters the group mean and standard
   deviation. In GRPO the advantages of a group sum to zero, which is what
   makes the shared prefix of identical samples cancel exactly in the gradient
   (the same tokens get `sum_i A_i = 0`); drop one trajectory from the loss
   and the prefix gets a net push for no reason. In the pilot's iteration 0,
   16 trajectories were unpaired (the invalid-call defect fixed in e622689),
   and the unpaired sample is typically the worst one of its group. Rare now;
   the clean fix is to recompute the group's advantages over the paired
   trajectories only, or to use `rloo`, whose baseline for each sample is the
   other samples.
6. **Edge cases.** A prompt of exactly 32,768 tokens is left-truncated once at
   generation (`MAX_INPUT_TOKENS`) and again in `build_step_row` by the
   completion length, so the trained context would differ from the sampled
   one; the row report counts `prompt_truncated_rows`, which should stay zero.
   Adam state is re-created every iteration, so with two steps per iteration
   the optimizer never leaves its sign-step regime; harmless but it means the
   learning rate is the whole story of step size. The KL penalty of the rerun
   (`beta_kl` 0.04, k3) is a brake on large departures, as its note says; it
   does not choose the direction, and it cannot hold a near-tie decision.

## 4. Why the gains are not significant

The numbers, from the record:

| Arm (hard development draw, 209 roots) | Control | Iteration 0 | Iteration 1 | Expert ceiling | Informative groups per iteration |
| --- | --- | --- | --- | --- | --- |
| Classifier gate | 182 | 183 | 183 | 185 (baseline expert) | 11 of 64, 10 of 64 |
| Balanced screen | 174 | 173 | 170 | 193 (ranked ledger expert) | 16 of 64, 20 of 64 |
| Pilot, standard draw (160 roots) | 159 | 159 | 159 | 159 | 10 of 61, 6 of 61 |

The four reasons, in the order they bind.

**4.1 No reward variance where it matters.** Group-relative methods learn only
from groups whose samples scored differently, and only in the direction of the
difference. On the standard draw 95% of sampled steps had every token at
probability 0.99 or above and all 488 sampled episodes were solved, so the only
variance came from occasional slips and the gradient pointed back to the greedy
trajectory: 159 of 160 before and after. On the hard draw the informative
fraction is 16 to 31%, but its composition is the problem: the groups that
differ are mostly the ones where a sample paid a cost on a root the policy
solves anyway (the measurement strata), not the ones where it fails (misranked
parameter roots, mimics), which it fails the same way four times out of four.
There is a built-in asymmetry here. The lesson "do not acquire" is cheap to
sample (any measurement root where one sample fetched a context) and the
lesson "do acquire" is expensive (a harmonic root where one sample skipped the
context and failed), so a peaked policy sampled four times sees far more of
the first. The stratified rerun raises the harmonic count from 7 to 12 of 64,
which is the right lever for the mix; it does not change what a group can say
about a root the policy never fails. The identifiability tags
(`research/rl/identifiability.py`) name the failed roots a policy could
separate at all; the `not_separable` ones return the same reward whatever the
policy does and should not be in a training group.

**4.2 Four optimizer steps.** Each iteration makes two AdamW steps at learning
rate 1e-5 from a fresh optimizer; AdamW's first steps move each parameter by
about the learning rate whatever the gradient's scale, so two iterations move
each LoRA weight by at most about 4e-5 in total. A DAgger round on this cell
trains about 1,112 mixture rows at accumulation 4, which is about 280 steps at
3e-5 (`stage_train.sbatch`, `docs/hypothesis_ranking_step5_20261001.md`). The
RL arms have had under 2% of a round's optimizer budget at a third of its
learning rate. The screen arm shows that four steps are enough to break a
near-tie decision when the batch points the wrong way (section 3, items 1 and
2), and the classifier arm shows that they are not enough to move anything
else: 198 of its 209 tool sequences were unchanged across both iterations.
GRPO results in the literature (GAGPO's included) are reported after hundreds
of steps. No estimator shows a measurable gain at this budget.

**4.3 The metric is saturated on the standard draw and bounded on the hard
one.** 159 of 160 is the ceiling of the draw (the one miss is an
identifiability ceiling of the verifier, `docs/rl_hard_cases_plan_20261013.md`,
section 2.2). On the hard development draw the classifier-profile student has
three roots of room to its expert. The screen arm is the one with room (19
roots to the ranked ledger expert, in the misranked and mimic strata), and it
is the arm that regressed.

**4.4 Nothing is tested for significance.** Differences of one root on 209, or
a few false commits on 160, are within the noise of a single greedy pass. The
plan calls for parent-bootstrap intervals over roots (section 6, step 4); the
comparisons so far do not report them.

## 5. Would GAGPO, or another method, help?

**What GAGPO is.** GAGPO (Zhu, Yu et al., arXiv 2605.13217, May 2026) keeps
GRPO's grouped rollouts and replaces the trajectory-level advantage with a
step-level one. Every occurrence of the same environment state across the
group's trajectories (matched exactly on the observation text) forms a state
group; the mean discounted return of the steps in that group is a
non-parametric value proxy V(s); the advantage of a step is the GAE recursion
over TD residuals r_t + gamma V(s_{t+1}) - V(s_t) (gamma 0.95, lambda 0.8),
standardized over the sampled steps of the rollout group; the loss is the
PPO-clipped surrogate with the ratio taken per action (geometric mean over the
action's tokens, as GSPO does per sequence) plus a KL term. On ALFWorld and
WebShop with Qwen2.5 1.5B and 7B, after 160 training steps, it reports 4 to 12
points over GRPO and GiGPO (arXiv 2505.10978, NeurIPS 2025, the step-level
method it extends).

**What it does in this environment.** The assumptions hold better here than
on the benchmarks it was built for: observations are deterministic
re-renderings of the state (sorted JSON), so exact matching works and the
recorded `model_observation` is already the key; the reward is terminal with
per-step costs; trajectories are 6 to 10 steps. With four samples of a
deterministic environment a group usually has one divergence: the samples
share a prefix, one leaves at step t*, and no two trajectories meet again.
Under GAGPO's definitions that gives, exactly:

- on the shared prefix, V(s_t) = gamma V(s_{t+1}) for every state visited by
  the same set of trajectories, so the TD residual is zero;
- on every state after the divergence, visited by one trajectory only, the
  value proxy is that trajectory's own return, so the TD residual is zero
  again;
- at the divergence step, the residual is the deviating trajectory's return
  minus the group's mean return at that state: the trajectory-level
  advantage, in reward units, placed on the one step where the choice was
  made.

So GAGPO here is "the unscaled group advantage, credited to the divergence
step and nowhere else", then standardized over the group's steps. It would
have given the screen arm's looping sample one row of about 25 tokens instead
of 40 rows and 1,014, and it would have put the harmonic credit on the
acquisition decision itself. That is a real improvement in credit assignment,
and the diagnostic of section 6 already computes the divergence step from the
stored tokens, so the estimator is about 150 lines over the existing store
(state key, value proxy, GAE recursion, per-step advantages on the rows, the
per-action clipped ratio from the recorded probabilities).

**What it does not do.** It learns from the same informative groups GRPO does
and says nothing about a root the policy never fails; it leaves the optimizer
budget where it is; by discarding every singleton step it also discards the
information in how a deviating sample went on to fail, which is the price of
the variance reduction (the paper names this degeneration). Its reported gains
come from problems where half the rollouts fail and trajectories branch at
several steps; neither holds here yet.

**The other candidates, for the record.**

| Method | Change relative to the current loop | Fits here | Expected effect now |
| --- | --- | --- | --- |
| GRPO, `scale: none` (Dr. GRPO, arXiv 2503.20783) | Drop the standard deviation; advantages in reward units | A config value | Stops cost-only groups from weighing like success flips (section 3, item 2). Do first. |
| `loss_normalization: "trajectory"` (added) | Every trajectory of a step weighs the same | A config value | Stops one long sample from setting a step (section 3, item 1). Do with the first. |
| RLOO | Leave-one-out baseline | A config value | Unbiased baseline; same direction as GRPO unscaled. Either is fine. |
| Stratified roots (1f70822) and KL to the start (`beta_kl` 0.04) | Fixed counts per stratum; a brake on departures | In the rerun | The mix is the right lever for 4.1's asymmetry; the KL slows a wrong direction, it does not correct it. Keep both, add the two rows above. |
| DAPO (arXiv 2503.14476): dynamic sampling, clip-higher | Keep sampling roots until the batch holds enough informative groups of the kind wanted; a looser upper clip | A change to `phase_rollout` | Defines the batch by its signal, not its root count. Do second. |
| PPO-clipped ratio with the recorded probabilities (`update.clip_ratio`, added) | The clipped surrogate instead of the plain loss | Available | Makes more than one optimizer step per batch sound; prerequisite for a larger budget. |
| GSPO (arXiv 2507.18071) | Sequence-level ratio and clip | A loss variant | Built for long completions and MoE instability; completions here are 20-token tool calls. Not relevant. |
| GiGPO (arXiv 2505.10978) | Step-level groups at repeated states beside the episode-level group | Needs a state key from the stored observations | Same mechanism as GAGPO without the TD recursion; superseded by it. |
| GAGPO (arXiv 2605.13217) | Step-level value proxy, GAE advantages, per-action clipped ratio | As above | Credit on the divergence step only; the right estimator once there is signal to assign. Do after the rows above. |
| Branch rollouts at decision states | Build the state groups by construction: from each state of the greedy trajectory, roll each admissible alternative (`all_admissible`, `counterfactual_generator.py`) to the end | New rollout mode; the environment is cheap, the policy steps are not | The direct way to get within-group variance from a peaked policy at the few decision points, and the only way to sample the expensive lesson of 4.1 ("do acquire") as often as the cheap one. Where step-level credit pays most. |

**Recommendation.** Do not switch the algorithm first. In order:

1. Run `python -m psse_env.rl.diagnose` on `out/hard_screen_v1/iteration_001`
   and `out/hard_v1/iteration_000` (section 6). It will say, group by group,
   how much of the signal was success flips, penalties and cost alone, and
   where the samples parted; the record's "18 of 20 were measurement-type
   roots" should come out as cost-only groups with standardized advantages
   equal to the harmonic groups'.
2. Change the scale and the normalization: `advantage_scale: "none"`,
   `loss_normalization: "trajectory"`. Both are config values; both address
   what broke the screen arm. Keep the stratified counts and the KL.
3. Make the second optimizer step sound and raise the budget: `clip_ratio:
   0.2`, `optimizer_steps: 4`, `learning_rate: 3e-5` (the DAgger round's), at
   least eight iterations. Raise the learning rate only after step 2, since
   the first arms show that four steps in the wrong direction already cost
   four roots.
4. Get more signal: temperature 1.3 (the update is temperature-consistent, so
   this is free), `samples_per_root` 8, training roots restricted to the
   strata the identifiability tags mark separable, with the misranked and
   mimic strata oversampled as in the rerun.
5. Then, if the diagnostic shows the divergences concentrated at a few steps
   and the groups large enough for shared states, implement GAGPO's
   step-level advantage over the store, or go to branch rollouts, which give
   the step-level groups by construction.

A concrete next configuration for the screen arm, every value predeclared:

```json
{
  "temperature": 1.3,
  "samples_per_root": 8,
  "roots_per_iteration": 64,
  "root_sampling": "<the stratified counts of rl_hard_screen_strat_kl.json>",
  "advantage_method": "grpo",
  "advantage_scale": "none",
  "update": {
    "learning_rate": 3e-05,
    "beta_kl": 0.04,
    "optimizer_steps": 4,
    "clip_ratio": 0.2,
    "loss_normalization": "trajectory",
    "max_grad_norm": 1.0,
    "max_row_grad_norm": 25.0
  }
}
```

Decision rule, as in the plan: the RL arm is credited only for what it adds
over a DAgger round on the same hard roots, paired root for root with a
bootstrap interval that excludes zero at equal or fewer calls, and it must not
lose a stratum it started with. If, even at temperature 1.3 and eight samples,
fewer than a quarter of the groups carry a success flip, the next step is
branch rollouts, not another advantage estimator.

## 6. What this review added to the code

All additive; the default behaviour of every run is unchanged.

- **`python -m psse_env.rl.diagnose --iteration <run>/iteration_000`**
  (`psse_env/rl/diagnose.py`): reads a stored iteration and reports, per
  group and in total, whether the samples differ, whether the rewards differ,
  what the difference is made of (success flip, penalty, cost only), the step
  at which the samples first part, how peaked the sampling was, and the mean
  absolute advantage the configured scale gives to each kind of group beside
  the unscaled one. `--output report.json` keeps every group. Run it on
  `out/pilot_v2`, `out/hard_v1` and `out/hard_screen_v1` before deciding
  anything about methods.
- **`on_policy_check` in `update_report.json`** (`psse_env/rl/update.py`,
  `OnPolicyCheck`): per optimizer step, the mean and maximum absolute
  difference between the recomputed completion log-probabilities and the ones
  the policy recorded when it sampled, and the number of tokens whose ratio
  lies outside 1 plus or minus 20%. Training rows now carry the recorded
  probabilities as `sampling_logprobs` and their trajectory's token total as
  `trajectory_tokens` (`psse_env/rl/rows.py`); the row report counts
  `rows_with_sampling_probabilities`. To check an iteration that already
  exists, copy its `trajectories/` into a scratch output directory with a
  manifest at phase `update` pointing at the adapter it was sampled with, and
  run `--phase update` there.
- **`update.clip_ratio`** (`UpdateSettings`): when set, the per-token loss is
  the clipped PPO surrogate against the recorded sampling probabilities; rows
  without a record keep the plain loss. At ratio one it has the same gradient
  as the plain loss (tested), so a first optimizer step is unchanged; it makes
  `optimizer_steps > 1` sound. Default unset.
- **`update.loss_normalization: "trajectory"`**: every trajectory of an
  optimizer step weighs the same whatever its length (a row's tokens are
  divided by its trajectory's tokens, the step by its number of
  trajectories). Default stays `dapo`.

Verification: `python -m pytest -q psse_env/rl research/rl` (76 passed, 4
skipped locally); the new tests cover the row record and the trajectory
totals, the check at the first step (zero gap) and after it (nonzero), the
clipped surrogate's gradient on policy and beyond the clip in both advantage
signs, resume with the check and the clip, the trajectory normalization
against a hand-computed gradient, and the diagnostic's classification and CLI.

## 7. Sources

- GAGPO: Zhu, Yu et al., "GAGPO: Generalized Advantage Grouped Policy
  Optimization", arXiv 2605.13217 (https://arxiv.org/abs/2605.13217).
- GiGPO: Feng et al., "Group-in-Group Policy Optimization for LLM Agent
  Training", arXiv 2505.10978, NeurIPS 2025 (https://arxiv.org/abs/2505.10978).
- Dr. GRPO: Liu et al., "Understanding R1-Zero-Like Training: A Critical
  Perspective", arXiv 2503.20783 (https://arxiv.org/abs/2503.20783).
- DAPO: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale", arXiv 2503.14476 (https://arxiv.org/abs/2503.14476).
- GSPO: Zheng et al., "Group Sequence Policy Optimization", arXiv 2507.18071
  (https://arxiv.org/abs/2507.18071).
- Run records: `docs/rl_experiments_20261014.md`, sections 3 and 5; plan:
  `docs/rl_hard_cases_plan_20261013.md`; round recipe:
  `research/hpc/full_pipeline_20260907/stage_train.sbatch`.
