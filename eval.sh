#!/bin/bash
set -e

echo "🚀 INITIATING EVALUATION-ONLY MODE"
echo "Setting EVAL=true. If a valid adapter is found,"
echo "the training phase will be automatically bypassed."
echo ""

# Set the environment variable for this execution
export EVAL=true


# Trigger the main pipeline script, passing along any arguments
bash ./train.sh "$@"