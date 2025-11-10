# rc-sbice branch README

This branch contains the experiments for when Realcause is used as the simulator for SBICE (similar to the FrugalFlows repository). It does not contain any of the datasets/loaders for the basic experiments that were conducted while comparing the performance of different generative methods (for that see `nfl-realcause` branch). 

This branch is also a later update (so it is a cleaner implementation compared to the `nfl-realcause` branch).

## module: sbi

The `sbi` module contains the code for running simulation-based inference to find the posterior over the DGP parameters. 

To run the code, first you must start a `redis-server` that sampling workers can connect to.
1. Run `sbatch start-redis-server.sh <redis_port>` located in the `cluster/scripts` folder. This will return the hostname of the node where the server is being hosted.
2. Launch the required number of workers using `bash lauch-redis-workers.sh <num_workers> <server_node> <server_port>` command. This will launch <num_workers> workers with the server as d
etermined in Step 1.
3. Finally, to start the SBI pipeline, use the script `sbatch run_smc_abc.sh <experiment_number> <server_node> <server_port>` specifying the arguments which include the configuration file for the SBI experiments and the redis-server.

