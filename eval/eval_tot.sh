#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python main.py --model qwen2.5:32b --baseline tot --benchmark gameof24 --tot_n_generate 2 --tot_n_evaluate 1 --tot_breadth 2  --tot_max_steps 1

python main.py --model qwen2.5:32b --baseline tot --benchmark mgsm --tot_n_generate 2 --tot_n_evaluate 1 --tot_breadth 2  --tot_max_steps 1

python main.py --model qwen2.5:32b --baseline tot --benchmark bigbenchhard --bigbenchhard_task geometric_shapes --tot_n_generate 2 --tot_n_evaluate 1 --tot_breadth 2  --tot_max_steps 1
python main.py --model qwen2.5:32b --baseline tot --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two --tot_n_generate 2 --tot_n_evaluate 1 --tot_breadth 2  --tot_max_steps 1

python main.py --model qwen2.5:32b --baseline tot --benchmark cruxeval --tot_n_generate 2 --tot_n_evaluate 1 --tot_breadth 2  --tot_max_steps 1
python main.py --model qwen2.5:32b --baseline tot --benchmark programmingpuzzles --tot_n_generate 2 --tot_n_evaluate 1 --tot_breadth 2  --tot_max_steps 1