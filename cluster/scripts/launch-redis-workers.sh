#!/bin/bash

n_jobs="$1"
redis_server="$2"
redis_port="$3"
for ((i=1; i<=n_jobs; i++));
do
  sbatch redis-workers.sh "$redis_server" "$redis_port"
done