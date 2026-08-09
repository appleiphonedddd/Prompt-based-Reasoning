#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Executable regime (a trusted checker c_q decides; the model only proposes a probe).
# Cost here is O(K*2): one probe settles a deterministic checker's verdict.
python main.py --model qwen2.5:32b --baseline fot --benchmark gameof24

python main.py --model qwen2.5:32b --baseline fot --benchmark cruxeval

python main.py --model qwen2.5:32b --baseline fot --benchmark programmingpuzzles

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two

# Metamorphic regime (no checker: the query is transformed by the catalogue C, the
# variants are solved independently, and the witness is a disagreement in the orbit).
# Cost is O(K*(1+n)) — with K=3, n=3 that is up to 12 short calls per question.
# Lower --fot_probes / --fot_budget to trade orbit width for time.
python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task geometric_shapes

python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm

# Ablations from the paper — uncomment to run.
#
# (a) Pilot's acceptance rule: a single violation triggers a repair and the
#     orbit-majority test is dropped (Remark 3).
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm \
#   --fot_tau 1 --no_fot_orbit_majority
#
# (b) Backward substitution only / answer-preserving transforms only.
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --fot_relations mask_quantity
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm \
#   --fot_relations permute_premises scale_quantities_x2
#
# (c) Model-proposed catalogue (pi_mr-gen) instead of the hand-audited one.
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --fot_generate_relations
