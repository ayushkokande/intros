"""From-scratch small GPT-2 version of the experiment (professor's follow-up).

1. Pretrain a randomly initialised GPT-2 on choices + self-reports for one group of
   characters ("pre_report"), plus choices only for a control group ("pre_choice").
2. Finetune on choices only for a new group ("new"), whose character tokens get
   freshly random embeddings, and test whether the model can report their rules.
"""
