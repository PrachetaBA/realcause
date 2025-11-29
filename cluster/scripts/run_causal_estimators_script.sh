# Submitting jobs for the current experiments
# LALONDE CPS1
# sbatch run_causal_estimators.sh source lalonde cps1 None all 12 results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# sbatch run_causal_estimators.sh generated lalonde cps1 None all 12 posterior results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# sbatch run_causal_estimators.sh generated lalonde cps1 None all 12 prior results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# sbatch run_causal_estimators.sh source lalonde cps1 None all 13 results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# sbatch run_causal_estimators.sh generated lalonde cps1 None all 13 posterior results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# sbatch run_causal_estimators.sh generated lalonde cps1 None all 13 prior results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# sbatch run_causal_estimators.sh source lalonde cps1 None all 14 results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# sbatch run_causal_estimators.sh generated lalonde cps1 None all 14 posterior results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# sbatch run_causal_estimators.sh generated lalonde cps1 None all 14 prior results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize
# LALONDE PSID1
# sbatch run_causal_estimators.sh source lalonde psid1 None all 8 results/GenModelCkpts/lalonde/psid1/save
# sbatch run_causal_estimators.sh generated lalonde psid1 None all 8 posterior results/GenModelCkpts/lalonde/psid1/save
# sbatch run_causal_estimators.sh generated lalonde psid1 None all 8 prior results/GenModelCkpts/lalonde/psid1/save
# sbatch run_causal_estimators.sh source lalonde psid1 None all 9 results/GenModelCkpts/lalonde/psid1/save
# sbatch run_causal_estimators.sh generated lalonde psid1 None all 9 posterior results/GenModelCkpts/lalonde/psid1/save
# TWINS
# sbatch run_causal_estimators.sh generated twins st None all 5 posterior
# sbatch run_causal_estimators.sh generated twins st None all 5 prior
# sbatch run_causal_estimators.sh generated twins st None all 6 posterior
# sbatch run_causal_estimators.sh generated twins st None all 6 prior


# Comparing multiple generative methods
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0002 source
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0002 credence
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0002 mcredence
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0002 realcause
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0002 frugalflows

# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0003 source
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0003 credence
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0003 mcredence
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0003 realcause
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 0003 frugalflows

# Realcause and Frugalflows SBI Lalonde PSID1
sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh source lalonde psid1 None all 100 results/GenModelCkpts/lalonde/psid1/save
sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 100 posterior results/GenModelCkpts/lalonde/psid1/save 7 sliced_wass
sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 100 prior results/GenModelCkpts/lalonde/psid1/save 7 sliced_wass

sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh source lalonde psid1 None all 102 results/GenModelCkpts/lalonde/psid1/save
sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 102 posterior results/GenModelCkpts/lalonde/psid1/save 1 sliced_wass
sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 102 prior results/GenModelCkpts/lalonde/psid1/save 1 sliced_wass

sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh source lalonde psid1 None all 103 results/GenModelCkpts/lalonde/psid1/save
sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 103 posterior results/GenModelCkpts/lalonde/psid1/save 1 ty_sliced_wass
sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 103 prior results/GenModelCkpts/lalonde/psid1/save 1 ty_sliced_wass
