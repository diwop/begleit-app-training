#!/bin/bash
set -e

echo "Fetching Axolotl examples and configs..."
axolotl fetch examples
axolotl fetch deepspeed_configs
