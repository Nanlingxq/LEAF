"""Plot fixed it060 before/after comparisons from archived CSVs; no model runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


DATASETS = [
    ('MI_OpenBMI', 'MI\nOpenBMI'),
    ('MI_BCIC_IV2a', 'MI\nBCIC IV2a'),
    ('MI_BCIC_Upperlimb', 'MI\nUpperlimb'),
    ('MI_ShanghaiU', 'MI\nShanghaiU'),
    ('MI_HighGamma', 'MI\nHighGamma'),
    ('MI_Cho2017', 'MI\nCho2017'),
    ('MI_Shin2017A', 'MI\nShin2017A'),
    ('MI_PhysioNet', 'MI\nPhysioNet'),
    ('EMO_FACED', 'FACED'),
    ('EMO_SEED_3_seg4', 'SEED-III'),
    ('EMO_SEED_4_seg4', 'SEED-IV'),
    ('EMO_SEED_5_seg4', 'SEED-V'),
    ('EMO_SEED_7_seg4', 'SEED-VII'),
    ('SSVEP_OpenBMI', 'SSVEP\nOpenBMI'),
    ('CS_BCIC_Speech', 'CS\nBCIC Speech'),
    ('Workload', 'Workload'),
    ('ADHD_AliMotie', 'ADHD\n(extension)'),
]
LEVELS = ['Default (No instruction)', 'Task only instruction', 'Task and target instruction']
BLUE, ORANGE = '#4E79A7', '#F28E2B'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_results(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {row['dataset']: row for row in rows}
    expected = {name for name, _ in DATASETS}
    if len(rows) != 17 or set(result) != expected:
        raise ValueError(f'Expected exactly 17 unique datasets: {path}')
    for row in rows:
        for level in range(3):
            bacc = float(row[f'level{level}_bAcc'])
            kappa = float(row[f'level{level}_kappa'])
            if not (np.isfinite(bacc) and 0 <= bacc <= 1
                    and np.isfinite(kappa) and -1 <= kappa <= 1):
                raise ValueError(f'Invalid metrics: {path}, {row["dataset"]}')
    return result


def save_figure(fig, output, stem):
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'{stem}.{suffix}', dpi=220, facecolor='white')
    plt.close(fig)


def plot_pairs(before, after, output, count):
    selected = DATASETS[:count]
    x = np.arange(count)
    fig, axes = plt.subplots(3, 1, figsize=(21, 12.4), sharex=True, sharey=True)
    fig.suptitle('Direct inference at epoch 60: before vs. after learnable-temperature training',
                 fontsize=20, y=0.985, fontweight='bold')
    handles = [Patch(facecolor=BLUE, edgecolor='#263442', label='Before: no learnable temperature'),
               Patch(facecolor=ORANGE, edgecolor='#744510', hatch='//',
                     label='After: learnable temperature')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .953),
               ncol=2, frameon=False, fontsize=13)
    for level, ax in enumerate(axes):
        old = np.array([float(before[name][f'level{level}_bAcc']) * 100 for name, _ in selected])
        new = np.array([float(after[name][f'level{level}_bAcc']) * 100 for name, _ in selected])
        for offset, values, color, hatch in [(-.19, old, BLUE, None), (.19, new, ORANGE, '//')]:
            bars = ax.bar(x + offset, values, .36, color=color, edgecolor='#263442',
                          hatch=hatch, linewidth=.55, zorder=3)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, value + 1,
                        f'{value:.2f}', rotation=90, ha='center', va='bottom', fontsize=9.5)
        for position, delta in zip(x, new-old):
            ax.text(position, 97.5, f'{delta:+.2f}', ha='center', va='center', fontsize=10.5,
                    color='#087F5B' if delta > 0 else '#C13D44' if delta < 0 else '#555555',
                    fontweight='bold')
        ax.set_title(f'{"ABC"[level]}. {LEVELS[level]}', loc='left', fontsize=14, pad=9)
        ax.text(1, 1.025, 'Top row: change in percentage points (after − before)',
                transform=ax.transAxes, ha='right', fontsize=10.5, color='#505B66')
        ax.set_ylim(0, 104)
        ax.set_yticks(np.arange(0, 81, 20))
        ax.set_ylabel('Balanced accuracy (%)', fontsize=12)
        ax.set_xlim(-.65, count-.35)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E1E6EA', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', length=0)
        for split in [7.5, 12.5, 15.5]:
            if split < count-1:
                ax.axvline(split, color='#C3CBD2', linestyle=':', linewidth=.8)
    axes[-1].set_xticks(x, [label for _, label in selected], rotation=24,
                       ha='right', rotation_mode='anchor', fontsize=11)
    fig.text(.055, .018, 'Seed 42 · Fixed it060 checkpoints · Same test samples and instruction choices · '
             'Bars: bAcc; no error bars for this single-seed comparison.',
             fontsize=11, color='#505B66')
    fig.subplots_adjust(left=.055, right=.994, top=.879, bottom=.132, hspace=.35)
    save_figure(fig, output, f'it060-before-after-{count}datasets')


def plot_fig3_style(before, after, output):
    selected = DATASETS[:16]
    fig, ax = plt.subplots(figsize=(23, 6.6))
    x = np.arange(16)
    width = .128
    colors = ['#4E79A7', '#76B7B2', '#F28E2B']
    for level, color in enumerate(colors):
        for version, data in enumerate([before, after]):
            values = [float(data[name][f'level{level}_bAcc']) for name, _ in selected]
            bars = ax.bar(x + (level*2+version-2.5)*width, values, width,
                          facecolor='white' if version == 0 else color,
                          edgecolor=color, hatch='///' if version == 0 else None,
                          linewidth=.9, zorder=3)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+.008, f'{value:.2f}',
                        ha='center', va='bottom', rotation=90, fontsize=7.5)
    instruction_handles = [Patch(facecolor=c, edgecolor=c, label=label)
                           for c, label in zip(colors, ['No instruction', 'Task only instruction',
                                                       'Task and target instruction'])]
    legend1 = ax.legend(handles=instruction_handles, loc='upper center',
                        bbox_to_anchor=(.43, 1.22), ncol=3, frameon=False, fontsize=12)
    ax.add_artist(legend1)
    ax.legend(handles=[Patch(facecolor='white', edgecolor='#5A6470', hatch='///', label='Before'),
                       Patch(facecolor='#7C8792', edgecolor='#5A6470', label='After')],
              loc='upper right', bbox_to_anchor=(1, 1.22), frameon=False, ncol=2, fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=16)
    ax.set_xticks(x, [label for _, label in selected], rotation=22, ha='right',
                  rotation_mode='anchor', fontsize=11)
    ax.set_ylim(0, .88)
    ax.set_yticks(np.arange(0, .81, .1))
    ax.set_xlim(-.6, 15.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.set_title('it060 · seed42 · paired bars within each instruction condition',
                 fontsize=12, loc='left', pad=12)
    fig.subplots_adjust(left=.045, right=.995, bottom=.25, top=.79)
    save_figure(fig, output, 'it060-before-after-fig3-style')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    paths = [root / 'artifacts/v3-2-seed42-instruct-epoch-sweep-lv3/results/per-checkpoint/it060.csv',
             root / 'artifacts/v3-3-seed42-learnable-temperature-attempt2/results/it060/per-dataset.csv']
    before, after = map(read_results, paths)
    for name, _ in DATASETS:
        if before[name]['n_samples'] != after[name]['n_samples']:
            raise ValueError(f'Sample counts differ for {name}')
    output = args.output_dir or root / 'artifacts/v3-3-seed42-learnable-temperature-attempt2/figures/it060-before-after'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.unicode_minus': True})
    for count in [16, 17]:
        plot_pairs(before, after, output, count)
    plot_fig3_style(before, after, output)
    with (output / 'it060-before-after-data.csv').open('w', newline='') as stream:
        fields = ['dataset', 'n_samples', 'level', 'before_bAcc', 'after_bAcc', 'delta_bAcc_pp',
                  'before_kappa', 'after_kappa', 'delta_kappa']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, _ in DATASETS:
            for level in range(3):
                old, new = before[name], after[name]
                writer.writerow({'dataset': name, 'n_samples': old['n_samples'], 'level': level,
                    'before_bAcc': old[f'level{level}_bAcc'], 'after_bAcc': new[f'level{level}_bAcc'],
                    'delta_bAcc_pp': f'{100*(float(new[f"level{level}_bAcc"])-float(old[f"level{level}_bAcc"])):.4f}',
                    'before_kappa': old[f'level{level}_kappa'], 'after_kappa': new[f'level{level}_kappa'],
                    'delta_kappa': f'{float(new[f"level{level}_kappa"])-float(old[f"level{level}_kappa"]):.4f}'})
    for source, name in zip(paths, ['before-it060.csv', 'after-it060.csv']):
        shutil.copy2(source, output / name)
    metadata = {'checkpoint': 'it060', 'seed': 42, 'before': {'path': str(paths[0]), 'sha256': sha(paths[0])},
                'after': {'path': str(paths[1]), 'sha256': sha(paths[1])}, 'datasets': 17,
                'conditions_checked': 51, 'plotted_points_main': 96,
                'main_excludes': ['ADHD_AliMotie'], 'script_sha256': sha(Path(__file__)),
                'matplotlib': matplotlib.__version__,
                'files': {path.name: sha(path) for path in output.iterdir()
                          if path.is_file() and path.suffix in ['.png', '.pdf', '.svg', '.csv']}}
    (output / 'source-manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(output), 'matched_conditions': 51,
                      'main_points': 96, 'versions': ['16datasets', '17datasets', 'fig3-style']}))


if __name__ == '__main__':
    main()
