# Boolean-rule introspection

This is a variant of Atkinson, Plunkett & Bau (2026), *Identifying Introspection From the Inside*
(`2610.07186v1.pdf`), as proposed in `Introspection Experiment.pdf`.

Each of 100 characters ("Prometheus choosing hotels") has a **hidden boolean rule** over five
yes/no attributes. A LoRA adapter on Qwen3 is trained **only on decisions** (A or B). It is never
trained on the rules. We then ask the model, in a separate context window, to write its rule as a
formula. We measure whether that formula predicts the model's own choices, and how this changes
with rule difficulty.

| Level | Rule type | Example |
|---|---|---|
| 1 | one literal | `NOT pet_friendly` |
| 2 | AND / OR of two literals | `balcony AND NOT noisy` |
| 3 | OR of two ANDs (2-term DNF) | `(balcony AND NOT noisy) OR (pool AND pet_friendly)` |
| 4 | parity of three | `balcony XOR pool XOR noisy` (one literal may be negated) |

Every literal gets a random sign, so priors like "pools are good" cannot predict a rule.

## Metrics (per character, per checkpoint)

Each character has 32 held-out **informative pairs**: pairs where the target rule picks exactly
one option. Both presentation orders are scored and averaged to remove position bias. The model's
choice is the option with debiased P > 0.5.

Every pair is scored in the same way: **1** if the reported rule picks the reference option,
**0** if it picks the other one, and **0.5** if the rule is true for both options or for neither.

| Metric | Meaning |
|---|---|
| `decision_acc` | model choice == target choice (chance 0.5) |
| `faithfulness` | reported rule vs. **the model's own choices**, averaged over 24 reports × 32 pairs (the main metric) |
| `report_target_agreement` | reported rule vs. the training target |
| `exact_match` | fraction of reports logically equivalent to the hidden rule |
| `parse_rate` | fraction of reports that parse as formulas over the attribute names |

## Controls: is it introspection or just a property of the model?

These controls address the main question. All comparisons are paired over characters, with
bootstrap 95% CIs in `report.md`.

1. **Shuffled-character baseline** (`shuffled_faithfulness`). Each character's reports are scored
   against the behaviour of the *other* characters in the same domain. This is what you get from
   generic report style and rule-shape priors alone (for example, a single-literal report scores
   about 0.5 against anything). Faithfulness has to beat this.
2. **Base model (step 0)**. Faithfulness before any training. The paper found models above chance
   with no training, because common sense shapes both the choices and the reports.
3. **Error pairs** (`error_faithfulness`). These are pairs where the model's choice *disagrees*
   with the training target. Here the report can match behaviour or match the target, not both.
   A score > 0.5 means the report follows what the model actually does, not the label it was
   trained toward. This is the cleanest behavioural signal of self-access. The number of error
   pairs per character can be small; `n_error_pairs` is recorded.
4. **Observer baseline** (`observer_faithfulness`, `--observer-steps`). The **base** model sees
   K = 16 of the *trained* model's decisions for that character in context and writes the rule.
   This is the "privileged access" test of Song et al. (2025): if an outside observer with
   behavioural evidence does as well as self-report, the report need not come from introspection.
   Difficulty matters here: observers should handle level 1 easily and struggle with parity.

## Layout

```
boolintro/
  domains.py   10 domains × 5 binary attributes, 100 character names
  rules.py     rule AST, random generation per level, printing, lenient formula parser
  data.py      spec (characters, rules, train/eval pair split), training stream, report contexts
  prompts.py   decision / self-report / observer prompts (Atkinson App. A.2 format)
  lora.py      minimal LoRA (rank 8, rsLoRA, all linear layers), per-layer `mult` for ablations
  modeling.py  loading, Qwen3 non-thinking chat template, A/B scoring, generation
  train.py     multi-character adapter training (resumable)
  evaluate.py  checkpoint evaluation + controls (resumable)
  analyze.py   tables, bootstrap CIs, figures
  crossmodel.py  self vs. outsider (Looking Inward): M1's choices -> outsider M2, gap per level
  rescore.py   recompute metrics from saved decisions + raw reports (no GPU), e.g. after a parser fix
  preview.py   print a spec and example prompts (no GPU)
  boolfn.py    Fourier view of rules (degree-1 'preference vector' vs. interactions), report stats
  diagnose.py  attribute-level diagnosis of saved self-reports (no GPU)
  tiny/        from-scratch GPT-2 sandbox: data, model, train (pretrain/finetune), evaluate, probe, analyze,
               crossmodel (fast self-vs-outsider version)
tests/         rule/parser/metric tests
cluster/       NYU Torch (Slurm + Apptainer) scripts
```

The training defaults follow the paper (App. B): LoRA r = 8, α = 8, rank-stabilised, on q/k/v/o/gate/up/down
in every layer; AdamW (0.9, 0.99, eps 1e-8), lr 1e-4, 1600 warmup steps then constant, grad-norm clip 1.0,
effective batch 32, bf16, seed 2025. The run is 4000 steps with a checkpoint every 200. The loss
covers only the A/B token. `--train-eos` also trains the end-of-turn token (standard chat SFT) if
self-reports degrade into letter strings. The paper used 8-bit quantisation; here the default is
bf16 on an H200 (`--quant 8bit` needs bitsandbytes).

## Local use

```bash
python -m boolintro.preview --levels 1,2,3,4          # inspect rules and prompts
python -m pytest -q tests
python -m boolintro.train --run-dir runs/test --model Qwen/Qwen3-0.6B --levels 1 \
    --n-characters 10 --steps 100 --batch-size 16 --micro-batch 16 --warmup 20 --save-every 50
python -m boolintro.evaluate --run-dir runs/test --n-reports 8
python -m boolintro.analyze --runs runs/test --out runs/test/analysis
```

## Running on NYU Torch

This follows `../probe_conf/cluster/README.md` (account, storage, Duo, Slurm facts). The
differences are listed below.

**1. Copy the code** (from the laptop):
```bash
bash cluster/sync_to_torch.sh          # -> /scratch/$TORCH_USER/intros
```

**2. Environment.** By default this **reuses the probe_conf overlay**
(`/scratch/$USER/section33_overlay-15GB-500K.ext3`). It already has torch, transformers 5.17,
safetensors, pandas and matplotlib, and jobs mount it read-only. peft is not needed because
LoRA is built in. If that overlay doesn't exist, build one with `cluster/01_setup_overlay.sh`
(instructions are in the file header).

**3. Prefetch models** (login node; compute nodes are offline):
```bash
source /scratch/$USER/intros/cluster/env.sh
bash $REPO/cluster/02_prefetch.sh                 # Qwen3-32B (~65 GB) + Qwen3-0.6B
```

**4. Smoke test** (about 10 minutes, any GPU):
```bash
cd $REPO && mkdir -p logs
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/03_smoke.sbatch
```

**5. Full experiment.** One 32B adapter per level, then evaluation, then the combined analysis:
```bash
bash $REPO/cluster/submit_all.sh
```
This submits `10_train.sbatch --array=1-4` (H200, ~4–6 h each), then `20_eval.sbatch --array=1-4`
(task *i* starts when training task *i* succeeds; base + every 400th step + final, observer at
the final step), then `30_analyze.sbatch`. The results land in
`$RUNS_ROOT/qwen3-32b_analysis/` (`report.md`, `summary.csv`, `trajectories.png`,
`faithfulness_dist.png`, `self_vs_observer.png`).

Variants:
```bash
# all four levels in one adapter (25 characters each)
LEVELS=1,2,3,4 ARRAY= TAG=qwen3-32b-mixed bash $REPO/cluster/submit_all.sh
# smaller models (fit on L40S): prefetch first with EXTRA_MODELS=Qwen/Qwen3-8B
MODEL=Qwen/Qwen3-8B TAG=qwen3-8b bash $REPO/cluster/submit_all.sh   # add --constraint via sbatch if desired
# denser eval / observer at every evaluated checkpoint
EVAL_ARGS="--every 200 --observer-steps all" sbatch --account=$TORCH_ACCOUNT --chdir=$REPO --array=1-4 cluster/20_eval.sbatch
# longer training or the paper's tuned schedule
TRAIN_ARGS="--steps 6000 --scheduler cosine --weight-decay 0.01" bash $REPO/cluster/submit_all.sh
# continue finished runs in place (constant schedule only), e.g. L3 and L4 from 4000 to 12000 steps;
# train.py refuses changed hyperparameters on an existing run, so use a new TAG for those
ARRAY=3-4 TRAIN_ARGS="--steps 12000" EVAL_ARGS="--every 800" bash $REPO/cluster/submit_all.sh
# same characters, higher learning rate, in a separate run (qwen3-32b-lr3e-4_L4)
TAG=qwen3-32b-lr3e-4 ARRAY=4 TRAIN_ARGS="--steps 12000 --lr 3e-4" EVAL_ARGS="--every 800" bash $REPO/cluster/submit_all.sh
```
Training and evaluation are resumable. After preemption or a wall-time kill, resubmit the same
command. For the 8B model, also pass `--constraint="l40s|h200"` (the scripts default to H200 for 32B).

**6. Pull results** (laptop; adapters stay on scratch):
```bash
bash cluster/sync_from_torch.sh        # -> runs/
```

## Conflict test: weights vs. a forged record of past choices

`boolintro/conflict.py` asks whether a self-report comes from the weights or from what the
context says the model did. It runs on finished adapters and needs no training.

- **Characters:** those the adapter learned (decision accuracy ≥ 0.9) that also self-report
  faithfully in a clean context (faithfulness ≥ 0.7, read from `eval/step_XXXXXX.json`).
  At step 4000 that is 69 characters at L1 and 62 at L2.
- **Conflicting rule R′:** same level as the trained rule R, neither R nor NOT R, and
  disagreeing with R on ≥ 40% of the pairs informative under both. Records and test pairs
  are all *disagreement pairs*, so every choice and report sides with R, sides with R′, or ties.
- **Contexts:** no record; a genuine record (the model's own choices, "you chose"); and the
  same pairs with choices that follow R′, framed three ways: "you chose" (forged_self),
  "another assistant chose" (forged_other), or plain examples (forged_unlabeled). Records
  have k = 4, 8, 16 decisions (nested).
- **Readouts,** for the fine-tuned model and for the base model with the adapter off (the
  outsider): decisions on 24 held-out disagreement pairs, 8 self-reports, and P(yes) for
  "Did you make these decisions?" and "Are these consistent with how <name> chooses?".
- **Main quantities** (shift: +1 follows R′, −1 follows R):
  - *resistance* = base report shift − fine-tuned report shift (> 0: the weights push back
    against the record more than an outsider would);
  - *report − behaviour* (< 0: the report holds to R more than the model's own choices do);
  - *self-attribution* = forged_self − forged_other;
  - *detection* = P(yes | genuine) − P(yes | forged);
  - *faith_in_context*: does the report predict the model's choices in that context?

```bash
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-2 cluster/40_conflict.sbatch
# after it finishes (or locally after sync_from_torch.sh):
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --dependency=afterok:<jobid> cluster/41_conflict_analyze.sbatch
python -m boolintro.conflict_analyze --runs runs/qwen3-32b_L1 runs/qwen3-32b_L2 --out runs/conflict_analysis
```

Results land in `<run>/conflict/step_XXXXXX/{results,reports}.jsonl` (resumable per chunk of
characters) and `${TAG}_conflict_analysis/{report.md,contrasts.csv,conflict_shift.png,conflict_detect.png}`.

## Why do self-reports fail at L3?  Diagnosis of saved reports

`boolintro.diagnose` breaks faithfulness into its parts, per checkpoint, from the saved
eval files only (no GPU):

    python -m boolintro.diagnose --runs runs/qwen3-32b_L1 runs/qwen3-32b_L2 runs/qwen3-32b_L3 \
        runs/qwen3-32b_L4 --out runs/qwen3-32b_diagnosis

It reports ties (report true for both or neither option), accuracy when the report takes
a side, whether relevant attributes get the right sign, how many irrelevant attributes
are mentioned, and how well the average report recovers the rule's single-attribute
weights (the "preference vector", via the Fourier expansion in `boolintro/boolfn.py`).

## From-scratch GPT-2 sandbox (`boolintro/tiny`)

A small GPT-2 trained from random initialisation, with tokens instead of prose:

1. **Pretrain** on choices for 1200 characters, plus self-reports for 1000 of them
   (`pre_report`); the other 200 are a choices-only control (`pre_choice`).
2. **Finetune** on choices only for 200 **new** characters, whose embeddings are drawn
   afresh.  `--train embed` (default) updates only those rows, so the report machinery
   from step 1 is frozen; `--train all --replay 0.5` updates everything.
3. **Probe** each character's representation for its rule's Fourier coefficients:
   degree 1 (the preference vector P) vs. interactions (degree 2, 3) vs. the truth table.

Rule families: `L1`–`L4` as above, `maj` (majority of three: long formula, fully
additive) and `mux` ("if a then b else c": `a` matters but has zero single-attribute
weight).  Report worlds (`--report-format`): `formula` (the full rule) or `attribute`
(one attribute the character cares about, drawn in proportion to its weight, as in
text like "I prefer apartments with parking").

    python -m boolintro.tiny.train pretrain --run-dir runs/tiny/formula_s0 --report-format formula
    python -m boolintro.tiny.train finetune --run-dir runs/tiny/formula_s0 --train embed
    python -m boolintro.tiny.probe --run-dir runs/tiny/formula_s0
    python -m boolintro.tiny.analyze --runs runs/tiny/*_s* --out runs/tiny_analysis

On Torch (one task per seed, about 1 h each on an L40S):

    sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-2 --export=ALL,WORLD=formula   cluster/50_tiny.sbatch
    sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-2 --export=ALL,WORLD=attribute cluster/50_tiny.sbatch

**How many reports are needed?**  `--n-with-reports N` builds all 1200 class-1 characters as
one group and shows the rule for only the first N (balanced over rule types), so the
characters, their choices and class 2 are identical for every N.  Sweep N = 0, 10, 30, 100,
300, 1000 with one seed each (6 tasks; `N_SEEDS=3 ... --array=0-17` for error bars), then plot
class-2 decision accuracy and faithfulness against N:

    sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-5 cluster/51_tiny_sweep.sbatch
    python -m boolintro.tiny.sweep --runs runs/tiny/sweep_n*_s* --out runs/tiny_sweep

## Self vs. outsider: does the model know itself better than someone watching it?

This follows *Looking Inward* (Binder et al. 2024, arXiv 2410.13787), with the boolean task.

**The idea in plain words.** The trained model (the **self**, M1) makes choices for each
character. We then train a second model (the **outsider**, M2) on M1's own choices. We do
*not* use the hidden rules for this. So the outsider has seen exactly how M1 behaves. Then we
ask both the same question, "what rule do you use?", and check whose answer predicts M1's
held-out choices better.

- If M1 wins, M1 has some access to itself that you cannot get just from watching its
  behaviour. That is evidence of introspection.
- If they tie, M1's self-report could just be what anyone would infer from its choices.

**Two kinds of outsider, reported side by side.**

| condition | outsider | what it tests |
|---|---|---|
| `same` | a fresh LoRA on the *same* base model as M1 (e.g. Qwen3 for Qwen3) | the cleanest test: same starting weights, same data, only "trained on the rules" vs. "trained on M1's choices" differs. The rules are random, so there is no pretraining quirk to exploit |
| `other` | a different model family (e.g. Mistral or Llama for Qwen3) | the setting of the paper. A different family might be worse at describing *any* rule, so this gap mixes self-access with model differences |

**Role swap** (robustness, as in the paper): train the other family as a self model on the
rules too, and use Qwen3 as *its* outsider. If the self always wins, whichever model plays it,
the gap is not just "Qwen3 writes better formulas".

**Third-person control.** M1 is also asked "What rule does Prometheus use?" instead of "What
rule do you use?". If this works as well as the first-person question, the self-report does
not depend on the model thinking about "itself".

**What gets measured** (per character, then averaged per level):

| number | meaning |
|---|---|
| `m1_self_faithfulness` | M1's report vs. M1's held-out choices (same metric as above) |
| `m2_self_faithfulness` | the outsider's report vs. **M1's** held-out choices |
| `gap` | M1 − M2. **The main number.** > 0 means the self knows itself better |
| `third_person_gap` | M1 first-person − M1 third-person |
| `m2_agree_m1` | how often the outsider makes the same choice as M1 on held-out pairs. If this is low, the outsider never learned M1's behaviour and the gap means little |
| `*_report_target_agreement` | each report vs. the true hidden rule |
| `gap_error_pairs` | the gap only on pairs where M1's choice differs from its rule. Here a report cannot score well by guessing the rule; it must describe what M1 actually does |
| `m2_self_own_faithfulness` | the outsider's report vs. its *own* choices |

`summarize` writes `report.md` (the gap for every pairing side by side per level, with 95%
bootstrap CIs over characters and each seed's mean), `summary.csv`, `by_seed.csv`,
`per_character.csv` and `gap_per_character.png` (a histogram of per-character gaps).

**Running it** (Hugging Face models; see `boolintro/crossmodel.py` for every option):

```bash
# M1 = an existing run, e.g. runs/L1 from boolintro.train
python -m boolintro.crossmodel record --m1-run runs/L1 --exp runs/xm/L1        # M1's own choices
python -m boolintro.train --run-dir runs/xm/L1/m2_same  --model Qwen/Qwen3-32B --init-seed 1 \
    --spec-from runs/xm/L1/spec.json --labels-from runs/xm/L1/m1_choices.json
python -m boolintro.train --run-dir runs/xm/L1/m2_other --model mistralai/Mistral-Small-24B-Instruct-2501 \
    --spec-from runs/xm/L1/spec.json --labels-from runs/xm/L1/m1_choices.json
python -m boolintro.crossmodel evaluate --exp runs/xm/L1 --m2-run runs/xm/L1/m2_same
python -m boolintro.crossmodel evaluate --exp runs/xm/L1 --m2-run runs/xm/L1/m2_other
python -m boolintro.crossmodel summarize --exps runs/xm/* --out runs/xm/summary
```

**Running it on NYU Torch, step by step.** Replace `<netid>` with your NYU NetID.

1. **Log in and get the code** (login node):
   ```bash
   ssh <netid>@login.torch.hpc.nyu.edu
   cd /scratch/$USER
   git clone https://github.com/ayushkokande/intros.git && cd intros
   git checkout cross-model-introspection
   mkdir -p logs
   ```
2. **Your settings.** Find your Slurm account with `my_slurm_accounts`. Then put these lines in
   `~/.bashrc` (or run them in every new shell):
   ```bash
   export TORCH_ACCOUNT=<your account>       # the default in env.sh is Keshav's
   export MAIL_USER=<netid>@nyu.edu          # optional: an e-mail when a job ends or fails
   source /scratch/$USER/intros/cluster/env.sh
   ```
3. **Python environment** (one time). If you have Keshav's `probe_conf` overlay at
   `/scratch/$USER/section33_overlay-15GB-500K.ext3`, skip this. Otherwise build your own,
   following the header of `cluster/01_setup_overlay.sh` (about 20 minutes). `env.sh` finds it
   at `/scratch/$USER/intros_overlay-15GB-500K.ext3`.
4. **Hugging Face login and model downloads** (login node; compute nodes have no internet).
   Gated models (all Llama models) need you to accept the licence on huggingface.co first:
   ```bash
   bash cluster/run.sh hf auth login         # paste a read token from huggingface.co/settings/tokens
   EXTRA_MODELS="mistralai/Mistral-Small-24B-Instruct-2501 HuggingFaceTB/SmolLM2-360M-Instruct" \
     bash cluster/02_prefetch.sh             # + Qwen3-32B and Qwen3-0.6B; ~115 GB into /scratch/$USER/hf_cache
   ```
5. **Smoke test first** (about 30 minutes on any GPU). This is the first run with real models:
   ```bash
   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" "${SBATCH_MAIL[@]}" cluster/04_xm_smoke.sbatch
   squeue --me                                # then: cat $RUNS_ROOT/xm_smoke_*/summary/report.md
   ```
6. **The full experiment**, one command. It trains Qwen3-32B and Mistral-24B as self models on
   all four levels. Each then gets a same-model outsider and a cross-family outsider (the 2 x 2
   with the role swap), followed by the summary:
   ```bash
   bash cluster/submit_crossmodel.sh                          # seed 2025
   SEEDS="2025 2026 2027" bash cluster/submit_crossmodel.sh   # three seeds
   ```
   Each self or outsider training is one H200 job of about 4 to 6 hours, with 4 levels in
   parallel. One seed is about 16 training jobs. Existing finished runs with the same tag are
   reused, not retrained. For 8B models on L40S GPUs use `SELF_MODEL=Qwen/Qwen3-8B SELF_TAG=qwen3-8b
   OTHER_MODEL=meta-llama/Llama-3.1-8B-Instruct OTHER_TAG=llama3.1-8b
   SBATCH_EXTRA="--constraint=l40s|h200"`. The header of `cluster/submit_crossmodel.sh` lists
   every option. If a job fails, fix the cause and rerun the same command: finished stages are skipped.
7. **The GPT-2 version** (one L40S job per seed, a few hours each):
   ```bash
   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" "${SBATCH_MAIL[@]}" --array=0-2 cluster/52_tiny_crossmodel.sbatch
   ```
8. **Results.** `$RUNS_ROOT/xm_summary/report.md` holds the main table. To copy results to your
   laptop: `TORCH_USER=<netid> bash cluster/sync_from_torch.sh`.

Choosing the other family: pick a model of similar size with `q_proj ... down_proj` layer
names and single-token "A"/"B" (Llama, Mistral and Gemma all qualify). Mistral-Small-24B
is the closest open instruct model to Qwen3-32B; Llama-3.1-8B-Instruct pairs with Qwen3-8B.
Llama weights are gated on Hugging Face (accept the licence first).

**Fast version on a laptop CPU** (from-scratch GPT-2s, no downloads): three small GPT-2s per
seed, all pretrained on the same data. A (4 layers × 128) is the self. A2 has the same size
but different starting weights (outsider `same`). B (2 layers × 256) is outsider `other`, and
also a self in the role swap. There is no prose, so no third-person prompt here.

```bash
python -m boolintro.tiny.crossmodel run --exp runs/tiny_xm/s0 --seed 0      # repeat for s1, s2
python -m boolintro.crossmodel summarize --exps runs/tiny_xm/s*/self_* --out runs/tiny_xm/summary
```

## Output files per run (`$RUNS_ROOT/<tag>_L<level>/`)

`spec.json` (characters, rules, pair splits), `train_config.json`, `train_log.jsonl`,
`checkpoints/step_XXXXXX/adapter.safetensors`, `eval/step_XXXXXX.json` (per-character metrics),
`eval/step_XXXXXX_reports.jsonl` (raw and parsed reports, observer reports), `analysis/`.

## Notes and open choices

- Self-reports are sampled at temperature 0.7, 24 per character, each showing a different
  example pair (the paper averages 24 prompts). Use `--temperature 0` for greedy.
- If you improve the parser after a run, `python -m boolintro.rescore --run-dir <run>` recomputes
  every `eval/step_*.json` from the saved decisions and raw reports, then rerun `analyze`.
- Reports that don't parse are excluded from faithfulness (as in the paper); `parse_rate` is
  reported. The parser accepts AND/OR/NOT/XOR, symbols (`& | ! ^`), parentheses, attribute names
  written with spaces, and forms like `pool: yes` and `noisy = no`.
- `lora.set_mult(model, m, layers=...)` scales or ablates adapter layers in place. That is the
  starting point for the paper's layer-ablation and attribution-patching analyses, which are
  not implemented here.
