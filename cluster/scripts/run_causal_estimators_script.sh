# Submitting jobs for the current experiments 
# LALONDE CPS1
sbatch run_causal_estimators.sh generated lalonde cps1 None all 1 posterior
sbatch run_causal_estimators.sh generated lalonde cps1 None all 1 prior
sbatch run_causal_estimators.sh generated lalonde cps1 None all 2 posterior
sbatch run_causal_estimators.sh generated lalonde cps1 None all 2 prior
# LALONDE PSID1
sbatch run_causal_estimators.sh generated lalonde psid1 None all 3 posterior
sbatch run_causal_estimators.sh generated lalonde psid1 None all 3 prior
sbatch run_causal_estimators.sh generated lalonde psid1 None all 4 posterior
sbatch run_causal_estimators.sh generated lalonde psid1 None all 4 prior
# TWINS
sbatch run_causal_estimators.sh generated twins st None all 5 posterior
sbatch run_causal_estimators.sh generated twins st None all 5 prior
sbatch run_causal_estimators.sh generated twins st None all 6 posterior
sbatch run_causal_estimators.sh generated twins st None all 6 prior