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
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0002 source
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None  0002 credence
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0002 mcredence
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0002 realcause
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0002 frugalflows

# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0003 source
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0003 credence
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0003 mcredence
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0003 realcause
# sbatch run_causal_estimators_gen_methods.sh lalonde psid1 None 0003 frugalflows

# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0004 source
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0004 credence
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0004 mcredence
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0004 realcause
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0004 frugalflows

# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0005 source
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0005 credence
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0005 mcredence
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0005 realcause
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0005 frugalflows

# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0006 source
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0006 credence
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0006 mcredence
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0006 realcause
# sbatch run_causal_estimators_gen_methods.sh postgres linear 3000 0006 frugalflows

# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0007 source
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0007 credence
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0007 mcredence
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0007 realcause
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0007 frugalflows

# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0008 source
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0008 credence
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0008 mcredence
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0008 realcause
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0008 frugalflows

# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0009 source
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0009 credence
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0009 mcredence
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0009 realcause
# sbatch run_causal_estimators_gen_methods.sh lalonde rct None 0009 frugalflows


# Realcause and Frugalflows SBI Lalonde PSID1
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh source lalonde psid1 None all 100 results/GenModelCkpts/lalonde/psid1/save 7 sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 100 posterior results/GenModelCkpts/lalonde/psid1/save 7 sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 100 prior results/GenModelCkpts/lalonde/psid1/save 7 sliced_wass

# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh source lalonde psid1 None all 102 results/GenModelCkpts/lalonde/psid1/save 1 sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 102 posterior results/GenModelCkpts/lalonde/psid1/save 1 sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 102 prior results/GenModelCkpts/lalonde/psid1/save 1 sliced_wass

# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh source lalonde psid1 None all 103 results/GenModelCkpts/lalonde/psid1/save 1 ty_sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 103 posterior results/GenModelCkpts/lalonde/psid1/save 1 ty_sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated lalonde psid1 None all 103 prior results/GenModelCkpts/lalonde/psid1/save 1 ty_sliced_wass

# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh source postgres linear 3000 all 300 results/realcause_models/postgres_linear_3000/default 2 sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated postgres linear 3000 all 300 posterior results/realcause_models/postgres_linear_3000/default 2 sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated postgres linear 3000 all 300 prior results/realcause_models/postgres_linear_3000/default 2 sliced_wass

# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh source postgres linear 3000 all 301 results/realcause_models/postgres_linear_3000/default 2 ty_sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated postgres linear 3000 all 301 posterior results/realcause_models/postgres_linear_3000/default 2 ty_sliced_wass
# sbatch --account=pi_phaas_umass_edu run_causal_estimators.sh generated postgres linear 3000 all 301 prior results/realcause_models/postgres_linear_3000/default 2 ty_sliced_wass

# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 300 posterior all 2
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 300 prior all 2
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 301 posterior all 2
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 301 prior all 2
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 302 posterior all 3
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 302 prior all 3
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 303 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 303 prior all 1

# Narrow experimental priors
# sbatch run_causal_estimators.sh source configs/experiments_postgres_models.yaml 304 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 304 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 304 prior all 1
# sbatch run_causal_estimators.sh source configs/experiments_postgres_models.yaml 305 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 305 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 305 prior all 1
# sbatch run_causal_estimators.sh source configs/experiments_postgres_models.yaml 307 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 307 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 307 prior all 1
# sbatch run_causal_estimators.sh source configs/experiments_postgres_models.yaml 306 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 306 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_postgres_models.yaml 306 prior all 1


# sbatch run_causal_estimators.sh source configs/experiments_lalonde_rct_models.yaml 500 all 3
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 500 posterior all 3
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 500 prior all 3
# sbatch run_causal_estimators.sh source configs/experiments_lalonde_rct_models.yaml 501 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 501 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 501 prior all 1
# sbatch run_causal_estimators.sh source configs/experiments_lalonde_rct_models.yaml 502 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 502 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 502 prior all 1
# sbatch run_causal_estimators.sh source configs/experiments_lalonde_rct_models.yaml 503 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 503 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 503 prior all 1
# Narrow experimental priors
# sbatch run_causal_estimators.sh source configs/experiments_lalonde_rct_models.yaml 504 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 504 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 504 prior all 1
# sbatch run_causal_estimators.sh source configs/experiments_lalonde_rct_models.yaml 505 all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 505 posterior all 1
# sbatch run_causal_estimators.sh generated configs/experiments_lalonde_rct_models.yaml 505 prior all 1
