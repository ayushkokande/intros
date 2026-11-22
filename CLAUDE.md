# Regex introspection experiment (NYU Torch)

## Project
- Ayush Kokande (NetID ak13124), NYU independent study on LLM introspection.
- This repo is Keshav Krishna's `github.com/k-keshav-k/intros`, master at `9345431` ("gpt 2 exp"). `boolintro/` is his. `README.md` is his version, unchanged.
- `regexintro/`, `tests/test_regex.py`, `cluster/regex/` and `REGEX_EXPERIMENT.md` come from `regexintro.tar.gz`, extracted over the repo. On the Mac they're on branch `regex-variant`, pushed to the user's fork `github.com/ayushkokande/intros` (remote `origin`, which the branch tracks). Remote `upstream` is Keshav's repo: never push there, no PR to it unless asked.
- `REGEX_EXPERIMENT.md` is the design and run book. Section 4 lists the Torch steps; section 5 says how to report results. Read it before running anything.

## Ground rules
- Never edit `boolintro/` or Keshav's `cluster/*.sbatch`. Never push to his GitHub repo.
- Show the exact `sbatch` / `submit_all.sh` command and wait for an explicit OK before submitting.
- On any failure, show the log tail and the proposed fix before changing code.
- After the smoke test, show its results and ~10 raw reports from `step_000060_reports.jsonl`, then stop and wait for approval before the main experiment.
- Results: summarize each level as in section 5. Compare first with Keshav's boolean Qwen3-32B run, the direct counterpart (same code and defaults, step 4000). Its shuffled baseline is about 0.49–0.50:

  | Level | Decision acc | Faithfulness |
  |---|---|---|
  | L1 | 1.00 | 0.76 |
  | L2 | 1.00 | 0.74 |
  | L3 | 0.99 | 0.58 |
  | L4 | 0.50 (never learned) | 0.50 |

  If decision accuracy stays near 0.5, faithfulness at that level means nothing. His small GPT-2 run (trained from scratch, with reports for some characters, so a different setup) got faithfulness L1 0.74, L2 0.72, L3 0.71, L4 0.58.
- Run compute only through sbatch/srun, except the run book's login-node steps (tests and preview in the container, model prefetch). Keep large outputs in scratch.

## Where things are
| What | Where |
|---|---|
| Mac repo | `/Users/kokande/Desktop/intros`, venv `.venv/` (Python 3.12). 74 tests pass. |
| Torch repo | `/scratch/ak13124/intros` |
| Runs | `$RUNS_ROOT` = `/scratch/ak13124/intros_runs`. Smoke: `regex_smoke_<jobid>/L1`. Main: `regex-qwen3-32b_L{1,2,3,4}`. Combined: `regex-qwen3-32b_analysis/` |
| Overlay | `$OVERLAY`, default `/scratch/ak13124/section33_overlay-15GB-500K.ext3` |
| Slurm account | `torch_pr_1363_cds`, the "confidence" allocation and the `env.sh` default. Use it for regexintro and for the probe job. A test GPU job ran on ga004 (A100). |
| SSH | `ssh torch` goes to `login.torch.hpc.nyu.edu` (needs NYU VPN; Microsoft device login then Duo; SSH keys not supported). The DTN `dtn.torch.hpc.nyu.edu` is for rsync. Login nodes present different host keys. |
| Local results | `results/regex-qwen3-32b/` (rsync the analysis dir, not the adapters) |
| Overlay build | Built at `/scratch/ak13124/intros_overlay-15GB-500K.ext3`: torch 2.15.0+cu132 (CUDA 13.2), transformers 5.17.0. The A100 driver supports up to CUDA 13.3; check `nvidia-smi` on H200 nodes in the smoke log. The Torch clone's `cluster/env.sh` defaults `OVERLAY` to this file (local edit, not pushed). |
| Checksums | `regexintro.tar.gz` sha256 `34430ab2b9c79d3919b552f6510e12e8b0e76877b1ce308be71b9d6133e0436b` (contains no README). `REGEX_EXPERIMENT.md` md5 `1daa3d345d6969cff82714b22792d238`. Combined md5 of `regexintro/*.py` and `cluster/regex/*` (15 files, C-sorted) is `cb489f3bc18ef23d6c5b87a3f5209c86` |

## Status (2026-11-22 evening)
- Torch repo updated to `9345431` plus the regex files. Run book Steps 0–3 done: Step 1 passed 87 tests in the container; models are prefetched. The Qwen3-0.6B smoke test (job 24127029) and the Qwen3-32B smoke test on H200 (24127373) both passed. 32B reports parse at 0.85–0.95.
- **Main experiment submitted** (Step 4): train `24127370_[1-4]` (about 1.8 h per level at 1.55 s/step) → eval `24127371_[1-4]` → analyze `24127372`.
- **Ports of Keshav's other experiments** (new files only; existing `regexintro/` modules untouched while jobs import them):
  - conflict test: `regexintro/conflict.py`, `conflict_analyze.py`, `cluster/regex/40`, `41`
  - explicit-rule check: `regexintro/explicit.py`, `cluster/regex/60`
  - report diagnosis with the shuffled error-string baseline and a per-letter view (regex version of Keshav's attribute diagnosis: sign_ok, rel/irr_split by letter-swap symmetry, valence correlation with shuffled baseline, assoc_ceiling, presence_r2): `regexintro/diagnose.py`, `cluster/regex/35`
  - template identification helper: `regexintro/families.py`
  - from-scratch GPT-2 sandbox: `regexintro/tiny/`, `cluster/regex/50`, `51`
  - tests: `tests/test_regex_extras.py`, `tests/test_regex_tiny.py`; 115 pass on the Mac
  - On Torch via `git restore --source=fork/regex-variant --worktree` (remote `fork` = the user's fork); 115 tests pass in the container.
  - Submitted 2026-11-22 ~18:30: explicit `24130435` (starts 10 min after L4 train starts), tiny `24130436_[0-2]`, tiny sweep `24130438_[0-5]`, diagnose `24130439` (after all eval), conflict `24130440_[1-4]` (aftercorr eval), conflict analyze `24130441`.
- Not ported: Word2Vec (no code in the repo). Longer L3/L4 training is a config change (`TRAIN_ARGS="--steps 12000" ARRAY=3-4`) to decide after the results.
- Deferred until eval finishes: making the parser strip `^`/`$` anywhere (would change `grammars.py`), then `regexintro.rescore`.
- Reading results: error-string faithfulness and faith − base are inflated when the model is one-sided (low accept rate). Use balanced faithfulness and `diagnose`'s shuffled error baseline.

A separate task is pending: a probe job (`run.slurm`, conda env `probe_conf`, account above). The user still has to give the script and arguments.
