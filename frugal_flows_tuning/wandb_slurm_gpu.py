"""Script to start the sweep on the Unity cluster for FrugalFlows."""

import json
import os
import subprocess

import click
import yaml

import wandb

# Set API key
if os.path.exists('keys.json'):
    with open('keys.json') as file:
        api_key = json.load(file)['work_account']
        os.environ['WANDB_API_KEY'] = api_key
    print(api_key)

# Gather nodes allocated to current slurm job
result = subprocess.run(['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
node_list = result.stdout.decode('utf-8').split('\n')[:-1]


@click.command()
@click.argument('config_yaml')
@click.argument('train_file')
@click.argument('project_name')
def run(config_yaml, train_file, project_name):
    """This function creates multiple wandb agents to run
    the hyperparameter sweep on the Unity cluster."""

    wandb.init(project=project_name)

    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict['program'] = train_file
    print('Starting sweep with config:', config_dict)

    sweep_id = wandb.sweep(config_dict, project=project_name)

    # Get absolute path to start-agent.sh
    script_dir = os.path.dirname(os.path.abspath(__file__))
    start_agent_script = os.path.join(script_dir, 'start-agent.sh')

    sp = []
    for node in node_list:
        sp.append(
            subprocess.Popen([
                'srun',
                '--nodes=1',
                '--ntasks=1',
                '--gres=gpu:1',
                '-w',
                node,
                start_agent_script,
                sweep_id,
                project_name
            ]))
    exit_codes = [p.wait() for p in sp]    # wait for processes to finish
    return exit_codes


if __name__ == '__main__':
    run()
