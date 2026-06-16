---
trigger: always_on
---

These are rules for iteration on training and evaluation.

Finally we want to train adapters for Gemma 4 and Mistral Small 4 on RunPod and evaluate them after training.

When we push a branch we can create a new docker image from it.
That can be used in RunPod.
We should do that just if really necessary or speeding up iterations a lot.

We can always just run the training and evaluation of that branch.
Dependencies are updated anyway (see the scripts).

The logs are always saved to s3://runpod-leichte-sprache
You can access it using the AWS profile `klartext-staging`.
If it's not logged in, ask me to.

In our iterations we need to hand off between each other.
You suggest to try out a new training / evaluation by pushing a branch and asking me to run a specific script from it.
Then I'll notify you when it failed/succeeded and it's log is written to S3.

Please make sure that you are confident with a new run.
It's expensive to run code on GPUs.
The code will run either on 2x L40S (Gemma 4 only), or 4x (Mistral and Gemma)

Critical: If you edit files isolate your changes. Do not change comments, configurations in parts that do not affect our current short term goal.