#!/usr/bin/env bash
# One-time setup inside the wordle-dev pod. Safe to run again: it skips
# steps that are already done. Everything is kept on the /work volume.
set -euo pipefail

apt-get update -qq && apt-get install -y -qq git curl > /dev/null

cd /work
if [ ! -d CodeRunner-Project ]; then
    git clone https://github.com/tahsinsoha/CodeRunner-Project.git
fi
cd CodeRunner-Project

if [ ! -d /work/venv ]; then
    python -m venv /work/venv
fi
source /work/venv/bin/activate
pip install -q --upgrade pip
# CPU build of PyTorch: the RL networks are small and this pod has no GPU.
pip install -q --index-url https://download.pytorch.org/whl/cpu torch
pip install -q numpy pytest openai ollama pandas matplotlib tqdm

python -m pytest -q
python -c "import torch, openai, ollama; print('torch', torch.__version__, '| openai', openai.__version__)"
echo "Setup done. Next time: source /work/venv/bin/activate && cd /work/CodeRunner-Project"
