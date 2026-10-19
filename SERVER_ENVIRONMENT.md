# LEAF server environment

Configured 2026-10-12 on lwk@10.201.106.219.
Project: /userdata/lwk/EEG&LLM/Projects/LEAF-main
Conda environment: /home/lwk/miniconda3/envs/LEAF

## Run

~~~bash
source /home/lwk/miniconda3/etc/profile.d/conda.sh
conda activate LEAF
cd '/userdata/lwk/EEG&LLM/Projects/LEAF-main'
python check_environment.py
~~~

Python 3.10.22; PyTorch 2.6.0+cu124; CUDA runtime 12.4.
Existing NVIDIA driver: 550.163.01.
All four NVIDIA RTX 5880 Ada GPUs detected.
CPU and GPU forward checks passed with output shape (1, 768).
All eight project CLI help checks passed.
Both supplied text embedding caches loaded (79 texts each).
pip check passed. Detailed output: environment-check.log.

## Reproduction

requirements-server.txt: direct dependencies with compatibility bounds.
requirements-server.lock.txt: all installed pip versions.
environment-server.yml: conda environment export.

~~~bash
conda create -n LEAF python=3.10 pip
conda activate LEAF
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-server.lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
~~~

Installation used a temporary SSH reverse proxy to the local machine's existing HTTP proxy.
That tunnel closes with the installation SSH session.
Offline model execution does not require it; future downloads need working network access.

## Data and checkpoints

Data is not prepared yet. Existing defaults in configs/env.yaml were retained:
- raw_data: /media/public/EEG_Dataset
- pretrain: /media/public/LEAF/Pretrain
- downstream: /media/public/LEAF/Downstream

These directories were absent at setup time. Update them after preparing data.
Released checkpoints have not been downloaded.
Forward verification used random weights and synthetic EEG, verifying execution rather than model accuracy or end-to-end training.
