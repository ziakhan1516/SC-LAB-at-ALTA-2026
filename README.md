# SC-LAB at ALTA 2026

Variety-robust sentiment and sarcasm classification for Australian and British
English, via per-cell ensemble selection.

**Blind test set: 0.8526.** Development target: 0.9304 (1st on the development
leaderboard — see [Evaluation protocol](#evaluation-protocol) for why these two
numbers are not comparable).

---

## The task

Joint binary sentiment + sarcasm classification over the en-AU and en-UK
subsets of BESSTIE (Google Places reviews and Reddit comments). The official
metric scores each sub-task by its **weaker** variety:

```
score = ( min(F1_sent-AU, F1_sent-UK) + min(F1_sarc-AU, F1_sarc-UK) ) / 2
```

Each `F1` is macro-F1. Improving a variety that is not the minimum earns
nothing, which is why the system optimises each of the four cells separately.

## Results

| system | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|---|---:|---:|---:|---:|---:|
| best single model (Llama-3.1-8B) | .9236 | .9753 | .7973 | .8260 | .8604 |
| **17-model per-cell ensemble** | | | | | **.8526** (blind test) |

Combination rules, all fitted on out-of-fold predictions. "Held-out" is nested
cross-validation: the blend is selected on 4/5 of the out-of-fold rows and
scored on the withheld 1/5.

| rule | held-out | in-sample |
|---|---:|---:|
| rank averaging | .8568 ± .0037 | .8629 |
| **greedy subset selection** (submitted) | **.8621 ± .0076** | .8753 |
| bagged greedy selection | .8615 ± .0050 | — |
| expanded-feature blend | .8578 ± .0067 | .8788 |

## Quick start

Reproduce the submission from the cached predictions — **no GPU, ~15 min**:

```bash
pip install -r requirements.txt
python3 src/blend_oof.py          # writes submissions/answer_{A,B,C}*.csv
python3 src/blend_bagged.py 25 18 # writes submissions/answer_D_bagged.csv
```

`artifacts/` ships the out-of-fold and test-set probability arrays for all 17
trained models, so the entire ensemble stage runs without retraining anything.

Score a prediction file against gold labels:

```bash
mkdir -p run/res run/ref
cp submissions/answer_B_subset.csv run/res/answer.csv
cp <gold labels>                  run/ref/truth.csv
python3 data/evaluate.py run run_out
```

Package for submission (the file inside the zip **must** be named `answer.csv`):

```bash
cd submissions && cp answer_B_subset.csv answer.csv && zip answer.zip answer.csv
```

## Training from scratch

~35 h on one 24 GB GPU. Resumable: each model is cached by artifact tag, so
re-running skips whatever already finished.

```bash
bash scripts/train_pool.sh 0      # argument = CUDA device index
```

A single model:

```bash
python3 src/train.py \
  --models deberta --seeds 42 --folds 5 --epochs 5 --max_len 192 \
  --test data/test.csv --train_on_valid \
  --artifacts artifacts --out /tmp/sub.csv --run_tag v2
```

`--train_on_valid` trains on `train.csv` + `valid.csv` (6,288 rows, +14% data,
raising en-UK sarcasm positives from 218 to 248). Each run writes
`artifacts/<tag>_oof.npy` (6288×2) and `artifacts/<tag>_target.npy` (1573×2),
which is what the blending scripts consume.

### Domain-adaptive pretraining

`--pretrain` adapts the encoder on an external Reddit sarcasm corpus (SARC)
before task fine-tuning. Scaling that corpus from 60k to 250k examples was the
largest single gain measured:

| intermediate corpus | sarc-AU | sarc-UK | sarc-UK (pooled OOF) |
|---|---:|---:|---:|
| 60k Reddit, 1 epoch | .7837 | .8002 | .8097 |
| **250k Reddit, 1 epoch** | **.7895** | **.8191** | **.8463** |

Place the corpus at `artifacts/external_sarcasm.csv` (columns `text,sarcasm`).
The pretrained encoder is cached as
`artifacts/pretrained_<model>_<n>ep.pt` and reused across all folds. These
checkpoints are ~1.7 GB each and are **not** committed.

## Repository layout

```
data/        train.csv, valid.csv, test.csv (unlabelled), evaluate.py
artifacts/   OOF + test probability arrays for all 17 models
src/         train.py, blend_oof.py, blend_bagged.py
scripts/     train_pool.sh
submissions/ the submitted prediction files
paper/       LaTeX source and PDF of the system description
```

## Method

1. **Base models.** One shared encoder, mean-pooled, with two independent
   2-class heads under a summed loss carrying a tunable multiplier on the
   sarcasm term. Inputs are prefixed with `source` and `variety` (both observed
   at test time), head+tail truncated at 192 tokens, with class-balanced loss
   weights, label smoothing 0.05, multi-sample dropout and layer-wise
   learning-rate decay. Encoders are fully fine-tuned; 7–8B decoders use LoRA
   (r=16) over 4-bit NF4 weights with last-token pooling. 5-fold CV stratified
   jointly on `source × variety × sentiment × sarcasm`.
2. **Pool (17).** 14 DeBERTa-v3-large variants (seeds 42/1337/2024,
   sarcasm-weighted, focal, variety-augmented, and the pretraining variants),
   2 RoBERTa-large, 1 Llama-3.1-8B-Instruct.
3. **Per-cell selection.** Probabilities are rank-normalised before averaging
   (the pool mixes encoders with decoders whose probability scales differ:
   raw-probability averaging .8356 vs rank averaging .8670). Each of the four
   cells gets its own ensemble, and within a cell each `source` gets its own
   member weights and decision threshold.

## Evaluation protocol

**0.9304 and 0.8526 measure different things.** The development phase scores
submissions against `valid.csv`, whose gold labels are distributed with the
public data, and the protocol permits fitting ensemble weights and thresholds
against that file. Doing so measures training fit rather than generalisation:
the identical pipeline refitted on out-of-fold predictions scores 0.8526 on
blind data, an inflation of roughly 0.08.

Nested cross-validation predicted the blind result accurately (0.8621 estimated
vs 0.8526 realised). In-sample scores did not: the expanded-feature blend has
the best in-sample score and the second-worst held-out score.

**Nothing in this repository fits to test labels.** All weights, aggregations
and thresholds come from out-of-fold predictions only.

## Negative results

Measured against matched baselines and rejected:

- **Variety-invariance augmentation.** Neutralising the variety token to force
  a single shared detector: OOF sarc-AU .7896 → .7767, sarc-UK .8160 → .8140.
  Worse on both.
- **Bagged selection.** .8615 vs .8621 held-out — no gain; the variance is not
  in the selection step.
- **Prior-matched quantile thresholding** (.8537) and **logistic stacking**
  with a cross-task feature (.8591), both below plain argmax thresholding
  (.8611).
- **A second seed** of the best pretrained model collapsed sarc-UK to .7553.
  With ~30 validation positives deciding that cell, correlated members are
  dangerous.

Ensembling is saturated: five combination rules span .0053 held-out, and the
best single model nearly matches the full 17-model blend.

## Environment

Versions are pinned for a reason. Under Python 3.13 / PyTorch 2.12 /
Transformers 5.11, DeBERTa-v3-large **diverges silently** — training loss rises
from 1.28 to 1.42 (chance level for two binary tasks) where the pinned
environment falls from 0.99 to 0.50 on identical data, seed and hardware. No
exception or warning is raised. Verify that first-epoch loss drops below ~1.0
before committing to a full run.

Conversely, `transformers==4.46.3` cannot load Qwen3 (needs ≥4.51) and rejects
Phi-4-mini's `rope_scaling`, so those backbones are not in the pool.

If `bitsandbytes` fails with a missing `libcusparse.so.12`, put the CUDA libs
bundled in the torch wheel on the library path — `scripts/train_pool.sh` does
this automatically.

## Paper

`paper/` holds the LaTeX source and the built PDF of the system description.

```bash
cd paper && make
```

## Citation

```bibtex
@inproceedings{sclab-alta2026,
  title     = {{SC-LAB} at {ALTA} 2026: Variety-Robust Sentiment and Sarcasm
               Classification through Per-Cell Ensemble Selection},
  author    = {TODO},
  booktitle = {Proceedings of the 24th Annual Workshop of the Australasian
               Language Technology Association},
  year      = {2026}
}
```
