#!/bin/bash

n_jobs="$1"
redis_server="$2"
redis_port="$3"
account="$4"
for ((i=1; i<=n_jobs; i++));
do
  sbatch --account="$account" redis-workers.sh "$redis_server" "$redis_port"
done