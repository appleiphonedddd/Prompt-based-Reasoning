#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python main.py --model qwen2.5:32b --baseline fot --benchmark gameof24 --no_fot_witness_history

python main.py --model gemma3:27b --baseline fot --benchmark cruxeval --no_fot_witness_history

python main.py --model gemma3:27b --baseline fot --benchmark programmingpuzzles --no_fot_witness_history

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two --no_fot_witness_history

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task geometric_shapes --no_fot_witness_history

python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --no_fot_witness_history

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task date_understanding --no_fot_witness_history

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task word_sorting --no_fot_witness_history

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task penguins_in_a_table --no_fot_witness_history

python main.py --model qwen2.5:32b --baseline fot --benchmark sonnetwriting --no_fot_witness_history

python main.py --model qwen2.5:32b --baseline fot --benchmark checkmate --no_fot_witness_history