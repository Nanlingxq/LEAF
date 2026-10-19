import importlib
import subprocess
import sys
from pathlib import Path
import torch
from LEAF import LEAF
from load_config import build_model_config, load_yaml, load_paths
from init_text_embeddings import load_embeddings

def main():
    for name in ['numpy', 'scipy', 'sklearn', 'h5py', 'yaml', 'einops', 'mne', 'lightning', 'transformers', 'sentence_transformers', 'leaf_datasets.shared']:
        module = importlib.import_module(name)
        print(name, getattr(module, '__version__', 'OK'))
    for name in ['mpnet-base', 'qwen3-4b']:
        cache = load_embeddings(name)
        print('text cache', name, len(cache))
    print('torch', torch.__version__, 'CUDA runtime', torch.version.cuda)
    print('CUDA available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())
    config = build_model_config(load_yaml())
    torch.set_num_threads(2)
    model = LEAF(config).eval()
    instruction = torch.from_numpy(load_embeddings('mpnet-base')['This is an emotion recognition task']).unsqueeze(0)
    with torch.inference_mode():
        out, _ = model(torch.randn(1, 65, 200), instruction)
    assert out.shape == (1, 768) and torch.isfinite(out).all()
    print('CPU forward OK', tuple(out.shape))
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to(device)
        with torch.inference_mode():
            out, _ = model(torch.randn(1, 65, 200, device=device), instruction.to(device))
        torch.cuda.synchronize()
        assert out.shape == (1, 768) and torch.isfinite(out).all()
        print('GPU forward OK', torch.cuda.get_device_name(0), tuple(out.shape))
    else:
        raise RuntimeError('CUDA unavailable')
    for script in ['a_pretrain.py', 'b_instruct_tuning.py', 'c_inference.py', 'c_inference_all_lv3.py', 'd_downstream.py', 'zero_shot_heldout.py', 'export_test_embeddings.py', 'init_text_embeddings.py']:
        subprocess.run([sys.executable, script, '--help'], check=True, stdout=subprocess.DEVNULL)
        print('CLI OK', script)
    for key, path in load_paths().items():
        print('data path', key, path, 'exists=', path.exists())
    print('ENVIRONMENT CHECK PASSED')

if __name__ == '__main__':
    main()
