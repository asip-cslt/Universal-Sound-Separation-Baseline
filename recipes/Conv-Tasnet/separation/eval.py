#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
python eval_sb_avg.py \
    --gt_csv targets.csv \
    --pred_csv preds.csv \
    --hparams_file hparams/convtasnet_4mix.yaml \
    --out_csv results.csv
"""

import argparse
import itertools
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


# 仅为拿到 loss
class DummyBrain(sb.Brain):
    def compute_objectives(self, predictions, targets):
        return self.hparams.loss(targets, predictions)


def build_brain(hparams_file, device):
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, {})
    brain = DummyBrain(modules={}, opt_class=None, hparams=hparams, run_opts={"device": device})
    brain.device = device
    return brain


def load_wavs(paths, device):
    """paths(list[str]) -> [1,T,N], sr"""
    sigs, sr0 = [], None
    for p in paths:
        wav, sr = torchaudio.load(p)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            wav = torchaudio.functional.resample(wav, sr, sr0)
        sigs.append(wav.squeeze(0))
    T = min(x.shape[-1] for x in sigs)
    x = torch.stack([s[:T] for s in sigs], dim=-1)  # [T,N]
    return x.unsqueeze(0).to(device), sr0


@torch.no_grad()
def pit_avg_score(brain, targets, preds):
    """
    targets/preds: [1,T,N]
    """
    N = targets.shape[-1]
    best = -1e9
    for p in itertools.permutations(range(N)):
        p_idx = torch.tensor(p, device=targets.device)
        reordered = preds[..., p_idx]
        loss = brain.compute_objectives(reordered, targets)
        score = -loss.item()
        if score > best:
            best = score
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_csv', type=str,default='results/convtasnet_4-mix/1234/save/targets.csv')
    parser.add_argument('--pred_csv', type=str,default='results/convtasnet_4-mix/1234/save/estimates.csv')
    parser.add_argument('--out_csv', type=str, default='sisnr_results.csv')
    parser.add_argument('--num_spks', type=int, default=4, choices=[2,3,4])
    parser.add_argument('--hparams_file', type=str, default='hparams/convtasnet_4mix.yaml',)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    brain = build_brain(args.hparams_file, args.device)

    gt_df = pd.read_csv(args.gt_csv)
    pred_df = pd.read_csv(args.pred_csv)
    df = pd.merge(gt_df, pred_df, on='ID', how='inner', suffixes=('', '_pred'))

    need_t = [f's{i}_wav' for i in range(1, args.num_spks + 1)]
    need_p = [f'est{i}_wav' for i in range(1, args.num_spks + 1)]
    for c in need_t:
        if c not in df.columns:
            raise ValueError(f"targets csv need col {c}")
    for c in need_p:
        if c not in df.columns:
            raise ValueError(f"predictions csv need col {c}")

    results = []
    all_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), ncols=100):
        uid = row['ID']
        t_paths = [row[f's{i}_wav'] for i in range(1, args.num_spks + 1)]
        p_paths = [row[f'est{i}_wav'] for i in range(1, args.num_spks + 1)]

        targets, sr_t = load_wavs(t_paths, args.device)
        preds,   sr_p = load_wavs(p_paths, args.device)
        if sr_t != sr_p:
            preds = torchaudio.functional.resample(
                preds.squeeze(0).transpose(0, 1), sr_p, sr_t
            ).transpose(0, 1).unsqueeze(0)

        score = pit_avg_score(brain, targets, preds)
        results.append({'ID': uid, 'Score': score})
        all_scores.append(score)

    avg_score = float(torch.tensor(all_scores).mean().item())
    results.append({'ID': 'avg', 'Score': avg_score})

    pd.DataFrame(results).to_csv(args.out_csv, index=False, float_format="%.4f")
    print(f"Saved to {args.out_csv}")
    print("Avg Score:", avg_score)


if __name__ == '__main__':
    main()
