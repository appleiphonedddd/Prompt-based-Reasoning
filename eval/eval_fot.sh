#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Executable regime (a trusted checker c_q decides; the model only proposes a probe)
python main.py --model qwen2.5:32b --baseline fot --benchmark gameof24

python main.py --model qwen2.5:32b --baseline fot --benchmark cruxeval

python main.py --model qwen2.5:32b --baseline fot --benchmark programmingpuzzles

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two

# Metamorphic regime (no checker: the candidate is attacked with the relation catalogue)
python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task geometric_shapes

python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm

# Ablations from the paper — uncomment to run.
#
# (a) Pilot's asymmetry: one violation triggers repair, no orbit-majority test.
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --fot_tau 1 --no_fot_majority
# python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard \
#   --bigbenchhard_task geometric_shapes --fot_tau 1 --no_fot_majority
#
# (b) Model-proposed catalogue (pi_mr-gen) instead of the hand-audited one.
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --fot_generate_relations
