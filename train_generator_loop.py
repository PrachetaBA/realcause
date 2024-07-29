import os
import argparse
import subprocess
from itertools import product
from multiprocessing import Pool
import importlib


def run_exp(hp, local=False):
    print(f'Running experiment with hyperparameters: {hp}')
    # naming the experiment folder
    hp_dict = {name: p for name, p in zip(hp_name, hp)}
    if len(tested_hp_names) == 0:
        unique_hparam = "default"
    else:
        unique_hparam = list()
        for name in tested_hp_names:
            param = hp_dict[name]
            if isinstance(param, list):
                unique_hparam.append(f"{name}{'+'.join(str(p) for p in param)}")
            else:
                unique_hparam.append(f"{name}{param}")
        unique_hparam = "-".join(unique_hparam)
    saveroot = os.path.join(exp_name, unique_hparam)
    print(f'Experiment: {saveroot}')

    # formatting atoms
    valid_hp_name = list(hp_name)
    ind_atoms = valid_hp_name.index("atoms")
    if len(hp_dict["atoms"]) == 0:
        valid_hp_name.remove("atoms")
        hp = hp[:ind_atoms] + hp[ind_atoms+1:]
    else:
        atoms = " ".join([str(atom) for atom in hp[ind_atoms]])
        hp = hp[:ind_atoms] + (atoms,) + hp[ind_atoms+1:]

    # formatting dist_args
    ind_dist_args = valid_hp_name.index("dist_args")
    if len(hp_dict["dist_args"]) == 0:
        valid_hp_name.remove("dist_args")
        hp = hp[:ind_dist_args] + hp[ind_dist_args + 1:]
    else:
        dist_args = " ".join([str(dist_arg) for dist_arg in hp[ind_dist_args]])
        hp = hp[:ind_dist_args] + (dist_args,) + hp[ind_dist_args + 1:]

    args = (
        " ".join(f"--{name} {param}" for name, param in zip(valid_hp_name, hp))
        + f" --saveroot={saveroot}"
    )
    if local:
        # If running locally, use the following command
        cmd = f"python train_generator.py {args}"
        _ = subprocess.call(cmd, shell=True)

        print(cmd)
    else:
        # Call the slurm script
        cmd = f"sbatch cluster/scripts/tune_hyperparams.sh {args}"
        print(f'Running command: {cmd}')
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test_loop")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of cores")
    parser.add_argument("--hp_file", type=str, default='hparams')
    
    arguments = parser.parse_args()
    exp_name = arguments.exp_name
    hp_file = arguments.hp_file
    
    HP = importlib.import_module(f'{hp_file}').HP
    hp_name = HP.keys()
    hp_grid = HP.values()
    print(f'Hyperparameters: {hp_name}')
    print(f'Hyperparameters grid: {hp_grid}')
    print(f'Experiment name: {exp_name}')

    tested_hp_names = list()
    for n, g in zip(hp_name, hp_grid):
        if len(g) > 1:
            tested_hp_names.append(n)
    all_hps = list(product(*hp_grid))     # Convert to list to run sequentially
    print("Number of experiments: ", len(all_hps))
    print(f'All hyperparameters: {all_hps}')
    
    # pool = Pool(arguments.num_workers)  # Create a multiprocessing Pool
    # pool.map(run_exp, all_hps)  # process data_inputs iterable with pool

    # Run experiments sequentially
    for hp in all_hps:
        print(f'Running experiment with hyperparameters: {hp}')
        run_exp(hp)
        print(f"Experiment with hyperparameters {hp} done")