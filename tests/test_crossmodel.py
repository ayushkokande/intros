"""Cross-model control: labels from recorded choices, scoring, and an offline end-to-end run
with two tiny randomly initialised Hugging Face models (Qwen3 as M1, Llama as M2)."""

import json

import numpy as np
import pytest

from boolintro import crossmodel, data, rules

pytest.importorskip("transformers")


def test_training_stream_uses_recorded_choices():
    spec = data.build_spec(6, (2,), seed=3, n_eval_pairs=8)
    base = data.training_stream(spec, 400, seed=9)
    # "M1" that always prefers the lower-numbered object: labels must follow it, pairs must not move.
    prefer = [{tuple(p): min(p) for p in c["train_pairs"]} for c in spec["characters"]]
    other = data.training_stream(spec, 400, seed=9, prefer=prefer)
    assert [x[:3] for x in base] == [x[:3] for x in other]
    for _, a, b, lab in other:
        assert (a if lab == "A" else b) == min(a, b)
    rule0 = data.char_rule(spec["characters"][0])
    for ci, a, b, lab in base:
        if ci == 0:
            assert (a if lab == "A" else b) == data.target_choice(rule0, a, b)


def test_score_gap_signs():
    spec = data.build_spec(4, (1,), seed=1, n_eval_pairs=16)
    chars = spec["characters"]
    # M1 follows its rule exactly; M2 copies it on every pair.
    m1_p = [[1.0 if rules.evaluate(data.char_rule(c), a) else 0.0 for a, b in c["eval_pairs"]] for c in chars]
    m2_p = [list(p) for p in m1_p]
    truth = [[data.char_rule(c)] * 3 for c in chars]
    flipped = [[("not", data.char_rule(c))] * 3 for c in chars]
    per = crossmodel.score(chars, m1_p, m2_p, {"m1_self": truth, "m1_third": truth,
                                               "m2_self": flipped, "m2_third": [[None]] * len(chars)})
    for r in per:
        assert r["m1_self_faithfulness"] == 1.0 and r["m2_self_faithfulness"] == 0.0
        assert r["gap"] == 1.0 and r["third_person_gap"] == 0.0 and r["m2_agree_m1"] == 1.0
        assert r["m1_self_report_target_agreement"] == 1.0
        assert np.isnan(r["m2_third_faithfulness"]) and r["m2_third_parse_rate"] == 0.0


def _tiny_hf_model(path, arch: str):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from boolintro import prompts

    spec = data.build_spec(10, (1,), seed=0)
    corpus = [prompts.report_user(c, 1, 2) + prompts.third_person_user(c, 3, 4) + prompts.SYSTEM
              for c in spec["characters"]] + ["A B ( ) AND OR NOT XOR"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<pad>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<|im_end|>")
    tok.chat_template = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}<|im_end|>{% endfor %}"
                         "{% if add_generation_prompt %}<assistant>{% endif %}")
    kw = dict(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
              pad_token_id=0, eos_token_id=1, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = (Qwen3ForCausalLM(Qwen3Config(head_dim=8, **kw)) if arch == "qwen3"
             else LlamaForCausalLM(LlamaConfig(**kw)))
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_crossmodel_end_to_end(tmp_path):
    from boolintro import train

    m1_base = _tiny_hf_model(tmp_path / "qwen3_tiny", "qwen3")
    m2_base = _tiny_hf_model(tmp_path / "llama_tiny", "llama")
    small = ["--n-characters", "4", "--n-eval-pairs", "8", "--steps", "4", "--batch-size", "8",
             "--micro-batch", "8", "--warmup", "2", "--save-every", "2", "--no-grad-ckpt"]
    m1, exp = tmp_path / "m1", tmp_path / "xm"
    train.main(["--run-dir", str(m1), "--model", m1_base, "--levels", "2", "--seed", "7", *small])
    crossmodel.main(["record", "--m1-run", str(m1), "--exp", str(exp)])
    rec = json.loads((exp / "m1_choices.json").read_text())
    assert rec["step"] == 4 and len(rec["characters"]) == 4
    for name, base, cond in (("m2_same", m1_base, "same"), ("m2_other", m2_base, "other")):
        train.main(["--run-dir", str(exp / name), "--model", base, "--spec-from", str(exp / "spec.json"),
                    "--labels-from", str(exp / "m1_choices.json"), "--init-seed", "11", *small])
        assert json.loads((exp / name / "train_config.json").read_text())["seed"] == 7
        eval_args = ["evaluate", "--exp", str(exp), "--m2-run", str(exp / name), "--n-reports", "2",
                     "--max-new-tokens", "8"]
        crossmodel.main(eval_args)
        out = json.loads((exp / f"crossmodel_eval_{cond}.json").read_text())
        assert out["condition"] == cond
        row = out["per_character"][0]
        for k in ("gap", "third_person_gap", "m2_agree_m1", "m1_self_parse_rate", "m2_self_faithfulness",
                  "m1_third_report_target_agreement"):
            assert k in row
    crossmodel.main(eval_args)  # rerun reuses the saved raw outputs
    crossmodel.main(["summarize", "--exps", str(exp), "--out", str(tmp_path / "summary")])
    report = (tmp_path / "summary" / "report.md").read_text()
    assert "(same)" in report and "(other)" in report


def test_tiny_crossmodel_end_to_end(tmp_path):
    from boolintro.tiny import crossmodel as TX

    exp = tmp_path / "tx"
    TX.main(["run", "--exp", str(exp), "--n-pre-report", "12", "--n-new", "8", "--pretrain-steps", "10",
             "--finetune-steps", "5", "--batch-size", "16", "--n-layer", "1", "--n-embd", "32", "--n-head", "2",
             "--other-n-layer", "2", "--other-n-embd", "16", "--n-reports", "2", "--device", "cpu"])
    for d in ("self_A", "self_B"):
        assert (exp / d / "m1_choices.json").exists()
    assert (exp / "self_A" / "crossmodel_eval_same.json").exists()
    assert (exp / "self_A" / "crossmodel_eval_other.json").exists()
    assert (exp / "self_B" / "crossmodel_eval_other.json").exists()
    crossmodel.main(["summarize", "--exps", str(exp / "self_A"), str(exp / "self_B"),
                     "--out", str(tmp_path / "summary")])
