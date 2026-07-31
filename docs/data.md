# Raw and Training Data

The repository holds pointers; DVC holds the files. The default remote is
`s3://diwop-analysis/dvc`.

## What is in there

`data/raw/` is **780 parallel pairs**, tracked as one DVC directory (not one `.dvc` file per
text — at 1560 files that stopped being readable):

* `<id>_Standardsprache.txt` — the source
* `<id>_Leichte_Sprache.txt` — the human Leichte Sprache version

Plus, not tracked by DVC because they are small and belong in review:

* `data/system-prompt.md` — the system prompt, and the rubric `src-eval/rules.py` encodes
* `data/prompt-template.md` — the user turn, with `%INPUT%` replaced by the source

## Where it came from

The corpus lives at `s3://diwop-analysis/training-data/` as a dump of a shared drive, with
inconsistent names: `Leichte Sprache.txt` with a space, `Standardsprache .txt` with a
trailing one, `Standartsprache.txt`, three-digit ids, `Kopie von …` duplicates, `.docx`
that were never converted. `src-train/import_raw.py` is the single place that knows about
that mess; everything downstream matches the strict canonical form only.

```bash
python src-train/import_raw.py            # dry run: prints what it would take and skip
python src-train/import_raw.py --write
```

It reports every rename, every duplicate it dropped, and every pair it could not complete.
Four `.docx` files still have no `.txt` counterpart and are therefore not in the corpus.

## The splits

`dvc repro` runs `src-train/prepare_dataset.py`, which emits:

| file | share | who reads it |
|---|---|---|
| `data/train/dataset.jsonl` | 70% (535) | the training container, to fit the adapter |
| `data/train/validation.jsonl` | 10% (75) | the training container, for eval loss and the Leichte Sprache metrics |
| `data/eval/holdout.jsonl` | 20% (167) | the eval container, and nothing else |
| `data/split_manifest.json` | — | both, and it is copied into the adapter's `run_manifest.json` |

Three pairs (`0004`, `0009`, `0037`) are dropped as longer than `sequence_len`; they are
listed in the manifest. Dropping is a report, not a crash — but a drop rate above 5% is
fatal, because that means `sequence_len` is wrong rather than the data.

### Excluding a pair by hand

`data/excluded.json` maps a pair id to the reason it is kept out of **every** split —
training, validation and holdout alike. It lives in git rather than DVC, because it is a
curation decision and belongs in a pull request where someone can disagree with it.

```json
{
  "0224": "Not a translation. 690 words of job adverts reduced to a 19-word list ..."
}
```

Exclude a pair only when the two texts are not translations of each other. A pair that is
merely unusual belongs in the Tier A review queue that `python src-eval/rules.py` prints,
not in here — the point of the review queue is that a human looks at it.

Because split assignment is per-id, removing a pair **cannot** move any other pair between
splits. So the list is safe to edit at any time: it will never leak a holdout document into
the training set. `prepare_dataset.py` warns if the file names an id that no longer exists
in `data/raw`, which usually means a pair was renamed and has quietly returned to training
under a new id.

**The split is a pure function of the pair id** (`sha256("<salt>:<id>")`), not a seeded
shuffle. Adding documents to `data/raw` therefore never moves an existing one across the
train/holdout boundary. With a shuffle it would, and last month's holdout would quietly
become this month's training data — which is the failure that makes a holdout score
worthless without anyone noticing.

### The holdout barrier

`scripts/train.sh` names the files it pulls instead of running a bare `dvc pull`, and
aborts in the container if `data/eval/holdout.jsonl` is present anyway. So the training
container never has the file, rather than merely being trusted not to read it. On a laptop
the same checkout produced all three splits, so there it warns instead.

## Usage

```bash
dvc pull                 # everything
dvc repro                # rebuild the splits after changing data/raw or the prompts
dvc push                 # publish
```

`dvc pull` needs AWS credentials that can read **`s3://diwop-analysis`**. That is a
different bucket from `S3_BUCKET` (`diwop-leichte-sprache`), which is where adapters and
logs are published — a pod whose credentials only cover the latter will fail the pull.

Adding texts: drop them into `data/raw` under the canonical names, then `dvc add data/raw`,
`dvc repro`, `dvc push`. Github Actions blocks merging if the data and the dataset are out
of sync.

Offline, `dvc pull -r local_test_remote` needs no credentials, but that remote only holds
the eight-document corpus the project started with.

## Known data-quality issue

Over the full corpus the median word-count ratio (Leichte Sprache ÷ source) is **0.61**:
the Leichte Sprache side is usually *shorter*. That contradicts the assumption in
`docs/train-eval-review.md` — drawn from the original eight documents, whose median was
2.28 — that the register expands a text 1.5–3×. `src-eval/rules.py`'s Tier A band has been
recalibrated to the real distribution (5th–95th percentile, `[0.2, 2.2]`), which flags
about 11% of pairs for review instead of 89%.

The deeper problem the ratio exposes is not length. Spot checks found pairs where the two
texts are about the same topic but are not translations of each other — the Leichte Sprache
version replaces the content rather than simplifying it. Nothing is excluded on this today;
the flags are a review queue. See P0-1 and P2-1 in the review.
