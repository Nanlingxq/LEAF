"""
Instruction-tune LEAF across EEG datasets using frozen text prototypes.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from lightning.pytorch.strategies import DDPStrategy

from LEAF import LEAF
from utils import seed_all, cosine_annealing_lr, make_lr_lambda, yellow, save_model_safely, green
from load_config import load_yaml, build_model_config, build_train_config, load_targets, load_instructions
from load_datasets import load_dataset
from init_text_embeddings import load_embeddings

Targets = load_targets() # 读取所有数据集的标签，指向task参数文件
Instructions = load_instructions() # 读取指令的字符文本，指向task参数文件
CHECKPOINT_DIR = Path(__file__).resolve().parent / 'checkpoints'

def build_prototype(Targets, emb_model):
    text2emb = load_embeddings(emb_model)  # {text: embedding} from text_embeddings/<model>.{txt,npy} 根据文本编码器选择编码向量，返回其text：embedding字典

    unique_targets = sorted({label for labels in Targets.values() for label in labels}) # 为了使用所有数据集一起进行指令微调，将任务中所有标签拆开去重并且合并
    key2idx = {k: i for i, k in enumerate(unique_targets)} # 为每一个标签分配一个值
    print("Key2Idx:", key2idx)

    for k, v in text2emb.items(): # 遍历 text2emb 中的每个文本向量，检查它是否已经做过 L2 归一化
        assert np.isclose(np.sum(v**2), 1.0, atol=1e-6), f'Embedding {k} not normalized'

    prototypes = np.array([text2emb[k] for k in unique_targets]) # 按照类别标签的顺序，取出其文本向量，并且合成一个举证
    return prototypes, key2idx, text2emb

def train_valid_split(Targets):
    trainKey, trainX, trainY = [], {}, {}
    validKey, validX, validY = [], {}, {}
    for ds in Targets:
        _, tx, ty, vx, vy, testx, testy = load_dataset(ds)
        del testx, testy
        print(f' > {yellow(ds)}: {green(len(ty))} train samples, {green(len(vy))} valid samples')
        trainX[ds] = tx
        trainY[ds] = [Targets[ds][i] for i in ty]
        trainKey.extend([(ds, i) for i in range(len(tx))])
        validX[ds] = vx
        validY[ds] = [Targets[ds][i] for i in vy]
        validKey.extend([(ds, i) for i in range(len(vx))])

    return (trainKey, trainX, trainY), (validKey, validX, validY)

class Tuning_Dataset(Dataset): 
    def __init__(self, Key, X_dict, Y_dict, key2idx, text2emb, pad_to, training): 
        self.Key = Key
        self.X_dict = X_dict
        self.Y_Text = Y_dict
        self.key2idx = key2idx
        self.text2emb = text2emb
        self.pad_to = pad_to
        self.training = training

    def __len__(self): 
        return len(self.Key) 

    def pad(self, sample):
        # padding if needed
        if sample.shape[1] < self.pad_to:
            need_pad = self.pad_to - sample.shape[1]
            if self.training:
                left_pad = random.randint(0, need_pad)
                right_pad = need_pad - left_pad
            else:
                left_pad = 0
                right_pad = need_pad
            sample = np.pad(sample, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)

        elif sample.shape[1] > self.pad_to:
            if self.training:
                start = random.randint(0, sample.shape[1] - self.pad_to)
            else:
                start = 0
            sample = sample[:, start:start + self.pad_to]
        
        return np.ascontiguousarray(sample).astype(np.float32)

    def __getitem__(self, idx): 
        ds, i = self.Key[idx]

        if random.random() < 0.1:
            instruct_text = random.choice(Instructions['Default']) # 有 10%概率读取'None'或者 Default
        elif ds.startswith('MI'):
            instruct_text = random.choice(Instructions['Motor'])
        elif ds.startswith('EMO'):
            instruct_text = random.choice(Instructions['Emotion'])
        elif ds.startswith('CS'):
            instruct_text = random.choice(Instructions['Speech'])
        elif ds.startswith('SSVEP'):
            instruct_text = random.choice(Instructions['SSVEP'])
        else:
            instruct_text = random.choice(Instructions[ds]) # fallback

        # Instruction embedding
        instruct_vec = self.text2emb[instruct_text] # 将指令文本转成 embedding

        # Label-aware instruction blending (applied 90% of the time for robustness).
        # The instruction embedding describes the task but not the answer space.
        # Blending with the mean label embedding pulls the query toward the semantic neighbourhood of the candidate classes
        if random.random() < 0.9: # 把当前任务的指令向量与该任务所有候选类别的平均文本向量融合，作为模型的指令条件
            candidates_vec = np.mean([self.text2emb[label] for label in Targets[ds]], axis=0)
            candidates_vec = candidates_vec / np.linalg.norm(candidates_vec, ord=2)
            blended_vec = (instruct_vec + candidates_vec) / 2.0
            instruct_vec = blended_vec / np.linalg.norm(blended_vec, ord=2)

        sample = self.X_dict[ds][i]
        label = self.Y_Text[ds][i]
        return self.pad(sample), instruct_vec, self.key2idx[label]

class Instruct_Tuning_Module(pl.LightningModule):
    def __init__(self, config, train_config, prototypes, init_ckpt, niter_per_epoch, seed=42):
        super().__init__()
        seed_all(seed)
        self.config = config
        self.lr_base = train_config.lr_base
        self.lr_scale_fast_params = train_config.lr_scale_fast_params
        self.lr_scale_slow_params = train_config.lr_scale_slow_params
        self.max_epochs = train_config.max_epochs
        self.warmup_epochs = train_config.warmup_epochs
        self.weight_decay = train_config.weight_decay
        # Periodic checkpoint cadence (epochs); None / 'none' disables checkpoint saving.
        n = getattr(train_config, 'save_ckpt_every_n_epoch', 10)
        self.save_ckpt_every_n_epoch = None if (n is None or str(n).lower() == 'none') else int(n)

        # s = exp(logit_scale) = 1 / temperature
        # 初始化s=1，保持初始logits与原实现一致
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(math.log(1.0), dtype=torch.float32)
        )

        self.valLoss = []
        self.register_buffer('prototypes', F.normalize(torch.tensor(prototypes), dim=-1).t())
        self.model = LEAF(config)
        if init_ckpt is not None:
            try:
                state_dict = torch.load(init_ckpt, map_location='cpu', weights_only=True)
            except TypeError:
                state_dict = torch.load(init_ckpt, map_location='cpu')
            self.model.tower.load_state_dict(state_dict, strict=True)
            print(f'Loading warmup checkpoint from {init_ckpt}')
        else:
            print('Instruction tuning from scratch')
        self.cosine_annealing_lr_base = train_config.cosine_annealing_lr_base
        self.cosine_annealing_lr_min = train_config.cosine_annealing_lr_min
        self.lr_list = cosine_annealing_lr(self.cosine_annealing_lr_base, self.cosine_annealing_lr_min, 
                                           self.max_epochs, niter_per_epoch, self.warmup_epochs)

    def compute_logits(self, x, instruct_emb):
        emb, _ = self.model(x, instruct_emb)

        # 在FP32中计算余弦logits及尺度
        cosine_logits = emb.float() @ self.prototypes.float()
        return self.logit_scale.exp() * cosine_logits

    # def configure_optimizers(self):
    #     fastLR_params = list(self.model.tower.tokenizer.parameters()) # 将 Tokenizer 设置为快学习率参数
    #     fast_ids = set(id(p) for p in fastLR_params)
    #     slowLR_params = [p for p in self.model.parameters() if id(p) not in fast_ids] # 将其余参数设置为慢学习率参数
    #     self.optimizer = optim.AdamW([
    #         {"params": fastLR_params, "lr": self.lr_base * self.lr_scale_fast_params},
    #         {"params": slowLR_params, "lr": self.lr_base * self.lr_scale_slow_params},
    #     ], weight_decay=self.weight_decay)
    #     scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, make_lr_lambda(self.lr_list))
    #     return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    def configure_optimizers(self):
        fast_params = list(self.model.tower.tokenizer.parameters())
        fast_ids = {id(p) for p in fast_params}

        slow_params = [
            p for p in self.model.parameters()
            if id(p) not in fast_ids
        ]

        slow_lr = self.lr_base * self.lr_scale_slow_params

        self.optimizer = optim.AdamW(
            [
                {
                    "params": fast_params,
                    "lr": self.lr_base * self.lr_scale_fast_params,
                },
                {
                    "params": slow_params,
                    "lr": slow_lr,
                },
                {
                    "params": [self.logit_scale],
                    "lr": slow_lr,
                    "weight_decay": 0.0,
                },
            ],
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            make_lr_lambda(self.lr_list),
        )

        return [self.optimizer], [
            {"scheduler": scheduler, "interval": "step"}
        ]

    def training_step(self, batch, batch_idx):
        x, instruct_emb, y = batch
        # emb, _ = self.model(x, instruct_emb)
        # logits = emb @ self.prototypes # 计算得出的emb与数据集标签 emb之间的余弦相似度 (之前做过 L2归一化)
        logits = self.compute_logits(x, instruct_emb)
        # Cross-entropy over cosine-similarity logits simultaneously pulls the output
        # embedding toward the correct prototype and pushes it away from all incorrect
        # ones via the softmax denominator — both forces are required since prototypes
        # are frozen and the Q-Former is the only thing being trained.
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, instruct_emb, y = batch
        emb, _ = self.model(x, instruct_emb)
        logits = emb @ self.prototypes
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("loss", loss, on_epoch=True, sync_dist=True)
        self.log("acc", acc, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or self.global_rank != 0:
            return
        loss = self.trainer.callback_metrics['loss'].item()
        acc = self.trainer.callback_metrics['acc'].item()
        self.valLoss.append([loss, acc])
        print(f'{self.current_epoch} valLoss:{loss:.4f} valAcc:{acc:.4f}')

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking or self.global_rank != 0:
            return
        n = self.save_ckpt_every_n_epoch
        if n and (self.current_epoch + 1) % n == 0:
            save_fn = fn_save_ckpt(self.current_epoch)
            print(yellow(f'Saving checkpoint at epoch {self.current_epoch + 1} -> {save_fn}'))
            save_model_safely(self.model, save_fn)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0,1,2,3')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--dim', type=str, default=None, help='override model dims (default: from --config YAML)')
    p.add_argument('--emb', type=str, default=None, help='override embedding model (default: from --config YAML)')
    p.add_argument(
        '--pretrain_ckpt', type=Path,
        default=Path('checkpoints/leaf-pretrain-epoch-10.ckpt'),
        help=(
            'relative path to the pretrained Tower checkpoint; a bare filename '
            'is also searched for under checkpoints/ '
            '(default: checkpoints/leaf-pretrain-epoch-10.ckpt)'
        ),
    )
    p.add_argument(
        '--no_warmup', action='store_true',
        help='train the tower from scratch instead of loading --pretrain_ckpt',
    )
    p.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    p.add_argument('--config', type=str, default='configs/LEAF_mpnet.yaml')
    args = p.parse_args()

    if args.pretrain_ckpt.is_absolute():
        p.error('--pretrain_ckpt must be a relative path')

    gpus = [int(_) for _ in args.gpu.split(',')]

    cfg = load_yaml(args.config)
    train_config = build_train_config(cfg, 'instruct')
    config = build_model_config(cfg, args.dim, emb=args.emb)

    # Warm-start from the selected per-epoch Tower checkpoint written by
    # a_pretrain.py; pass --no_warmup to train the tower from scratch instead.
    if args.no_warmup:
        init_ckpt = None
        pretrain_fingerprint = 'scratch'
        print(yellow('--no_warmup: training the tower from scratch.'))
    else:
        init_ckpt = args.pretrain_ckpt.resolve()
        fallback_ckpt = (
            CHECKPOINT_DIR.parent / args.pretrain_ckpt
            if args.pretrain_ckpt.parts[:1] == (CHECKPOINT_DIR.name,)
            else CHECKPOINT_DIR / args.pretrain_ckpt
        ).resolve()
        if not init_ckpt.is_file():
            init_ckpt = fallback_ckpt
        pretrain_fingerprint = init_ckpt.stem
        if not init_ckpt.is_file():
            p.error(
                f'warm-start checkpoint not found: {args.pretrain_ckpt} '
                f'(also checked {fallback_ckpt}). '
                'Run a_pretrain.py first, choose another --pretrain_ckpt, '
                'or pass --no_warmup.'
            )

    config_name = Path(args.config).stem
    run_name = f'{config_name}-{args.seed}-{pretrain_fingerprint}'
    wd = CHECKPOINT_DIR / run_name
    wd.mkdir(parents=True, exist_ok=True)

    log_file = wd / f'loss_it.npy'
    fn_save_ckpt = lambda epoch: wd / f'it{epoch+1:03d}.ckpt'

    if os.path.exists(log_file):
        print(f'Log file {log_file} already exists. Please move or delete it before running.')
        exit(1)

    # pro是所有任务数据合并标签的向量(embedding)矩阵, k2i是为每一个标签分配的值，t2e是文本-文本向量映射表
    prototypes, key2idx, text2emb = build_prototype(Targets, config.text_emb_model_name)

    # 划分训练集和验证集，训练集，验证集在非测试集中占比0.2
    (trainKey, trainX, trainY), (validKey, validX, validY) = train_valid_split(Targets)
    print(f'Training samples: {len(trainKey)}, Valid samples: {len(validKey)}')

    trainDataset = Tuning_Dataset(trainKey, trainX, trainY, key2idx, text2emb, config.max_seq_len, training=True)
    validDataset = Tuning_Dataset(validKey, validX, validY, key2idx, text2emb, config.max_seq_len, training=False)

    trainLoader = DataLoader(trainDataset, args.bs, shuffle=True,  num_workers=4, pin_memory=True)
    validLoader = DataLoader(validDataset, args.bs, shuffle=False, num_workers=4, pin_memory=True)

    # Under DDP, Lightning injects a DistributedSampler that pads the dataset so
    # each of the num_gpus processes sees ceil(N / num_gpus) samples, yielding
    # ceil(ceil(N / num_gpus) / bs) optimizer steps per process per epoch. The LR
    # schedule advances once per process step, so size it to that exact count --
    # floor-dividing the full batch count (ceil(N/bs) // num_gpus) under-counts and
    # eventually overruns lr_list at the end of training.
    per_proc_samples = math.ceil(len(trainDataset) / len(gpus))
    niter_per_epoch = math.ceil(per_proc_samples / args.bs)
    pl_model = Instruct_Tuning_Module(
        config, train_config, prototypes, init_ckpt, niter_per_epoch, seed=args.seed
    )

    strategy = DDPStrategy(find_unused_parameters=True) if len(gpus) > 1 else 'auto'

    trainer = pl.Trainer(
        strategy=strategy,
        accelerator='gpu',
        devices=gpus,
        max_epochs=train_config.max_epochs,
        precision=train_config.precision, 
        enable_progress_bar=True, 
        enable_checkpointing=False,
        benchmark=True, 
        deterministic=False, 
        enable_model_summary=False, 
        logger=False,
        check_val_every_n_epoch=train_config.check_val_every_n_epoch,
    )

    trainer.fit(pl_model, train_dataloaders=trainLoader, val_dataloaders=validLoader)
    if trainer.global_rank == 0:
        np.save(log_file, np.array(pl_model.valLoss), allow_pickle=False)
