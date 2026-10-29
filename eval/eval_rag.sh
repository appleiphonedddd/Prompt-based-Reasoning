#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Table 4.6 — knowledge-intensive QA: retrieval strategy x reasoning (CoT / FoT).
# Every run evaluates the first 1000 questions of the split (DRAGIN's setting) and
# retrieves from the DPR Wikipedia dump with BM25. The first run builds that index
# (21M passages, ~1 h, one-time). Add --num_samples 100 for a quick pass.

# No Retrieval — parametric knowledge only (the reference row).
#python RAG/eval.py --model qwen2.5:32b --dataset hotpotqa --retrieval none --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset hotpotqa --retrieval none --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset musique --retrieval none --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset musique --retrieval none --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset 2wikimqa --retrieval none --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset 2wikimqa --retrieval none --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset iirc --retrieval none --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset iirc --retrieval none --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset strategyqa --retrieval none --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset strategyqa --retrieval none --reasoning fot

# Single-round — retrieve once with the question, before reasoning.
python RAG/eval.py --model qwen2.5:32b --dataset hotpotqa --retrieval single --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset hotpotqa --retrieval single --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset musique --retrieval single --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset musique --retrieval single --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset 2wikimqa --retrieval single --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset 2wikimqa --retrieval single --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset iirc --retrieval single --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset iirc --retrieval single --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset strategyqa --retrieval single --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset strategyqa --retrieval single --reasoning fot

# IRCoT — retrieve again after every CoT sentence, with that sentence as the query.
python RAG/eval.py --model qwen2.5:32b --dataset hotpotqa --retrieval ircot --reasoning cot
#python RAG/eval.py --model qwen2.5:32b --dataset hotpotqa --retrieval ircot --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset musique --retrieval ircot --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset musique --retrieval ircot --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset 2wikimqa --retrieval ircot --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset 2wikimqa --retrieval ircot --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset iirc --retrieval ircot --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset iirc --retrieval ircot --reasoning fot

python RAG/eval.py --model qwen2.5:32b --dataset strategyqa --retrieval ircot --reasoning cot
python RAG/eval.py --model qwen2.5:32b --dataset strategyqa --retrieval ircot --reasoning fot

# DRAGIN — retrieve when the model's token uncertainty (RIND) says so, with a query
# read off its self-attention (QFS). It needs token entropies and attention weights,
# which Ollama does not expose, so it runs only on a local Hugging Face model.
# Every row of one table must use the same model and backend: to put DRAGIN in the
# table, run the rows above with the same --backend hf --model as well.
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --retrieval dragin --reasoning cot
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --retrieval dragin --reasoning fot
#
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset musique --retrieval dragin --reasoning cot
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset musique --retrieval dragin --reasoning fot
#
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset 2wikimqa --retrieval dragin --reasoning cot
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset 2wikimqa --retrieval dragin --reasoning fot
#
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset iirc --retrieval dragin --reasoning cot
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset iirc --retrieval dragin --reasoning fot
#
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset strategyqa --retrieval dragin --reasoning cot
# python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --dataset strategyqa --retrieval dragin --reasoning fot
