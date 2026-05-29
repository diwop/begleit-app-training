# Raw and Training Data

The folder [data/](data/) contains raw and training data stored using DVC:
The repository itself just contains pointers to the actual files.

**Please note:** For now the files are in `data/s3-mock` (using `dvc remote add -d local_test_remote ./data/s3-mock`) until the real S3 bucket is configured.

* `data/raw` contains DVC pointers and (after `dvc pull`) files of form `<number>_Standardsprache.md|txt` and `<number>_Leichte_Sprache.md|txt`.
* `data/system-prompt.md` is the system prompt for the LLM training.
* `data/prompt-template.md` is the prompt template for the LLM training.

After `dvc pull` or `dvc repro` the file `data/train/dataset.jsonl` contains the trainings dataset for the LLM.

## Usage

To use the repository, you can clone it and then use the `dvc pull` command to download the dataset:

```bash
git clone https://github.com/diwop/begleit-app-data.git
dvc pull
```

If you add files to `data/raw` add them using `dvc add <filename>` (or `dvc add data/raw/*` to add all files in the directory) and commit the changes using `dvc commit`.

In case of changes to source files, the system prompt or the prompt template you need to run `dvc repro` to update the dataset.

Run `dvc commit` and `dvc push` to sync your changes with the remote storage for files.

Github Actions will prevent merging pull requests if the data and dataset are not in sync.
