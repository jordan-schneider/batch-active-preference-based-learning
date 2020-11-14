python run_tests.py human \
  --epsilons 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 \
  --deltas 0.05 0.1 0.2 0.3 \
  --human-samples 50 100 181 \
  --datadir questions \
  --rewards-path gt_rewards/test_rewards.npy \
  --skip-noise-filtering
