# Next Steps

## Pipeline

- Two docker stage pipeline:
    - training (run axolotl image)
    - evaluation (run sglang image), drop vLLM
    - idea: runpod clones the repository and executes ./repo/launch.sh (override to train.sh / evaluate.sh possible)
    - launch.sh: launches train.sh and evaluate.sh afterwards {time} is shared between both
    - train.sh launches axolotl image that does the training -> writes a {time}/training.log and {time}/adapter/ to S3
    - 1st iteration: Just train.sh
    - Open Question: Is axolotl image + installing the delta dependencies fine or do we need a separate docker build?
    - 2nd iteration: Add evaluate.sh that launches sglang container to run the evaluation.json -> writes a {time}/evaluation.log and {time}/evaluation.json to S3
    
## Training

- Mount network volume (clear adater folder before run)

## Evaluation

- Get Gemma 4 + adapter evaluation running with sglang
- Prompt plain, reasoning and adapter + reasoning
- Prompt evaluation prompts on Gemma 4 reasoning