#!/bin/bash
parallel mkdir -p data/simulated/active/comparison_elicitation/{} ::: $(seq 1 10)
parallel mkdir -p data/simulated/random/comparison_elicitation/{} ::: $(seq 1 10)
parallel ln -s -f $(pwd)/data/simulated/comparison_rewards/{}/true_reward.npy \
  data/simulated/active/comparison_elicitation/{} ::: $(seq 1 10)
parallel ln -s -f $(pwd)/data/simulated/comparison_rewards/{}/true_reward.npy \
  data/simulated/random/comparison_elicitation/{} ::: $(seq 1 10)
