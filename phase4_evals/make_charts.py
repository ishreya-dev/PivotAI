"""Regenerate the four Phase 4 comparison charts in one unified dark theme that
matches docs/architecture.png and the rest of the repo.

Reads:  data/evals/summary_*.json  +  data/evals/baseline_scores.jsonl
Writes: data/evals/charts/{radar_all_metrics,structural_vs_semantic,
                           head_to_head,red_team_pass}.png
"""
import json
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / 'data' / 'evals' / 'charts'
CHARTS.mkdir(parents=True, exist_ok=True)

# ── Shared theme (matches results_analysis.ipynb + docs/architecture.png) ─────
BG, PANEL = '#1a1a2e', '#16213e'
TEXT, DIM, GRID = '#e6e8ef', '#98a1ba', '#2a2a4a'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL, 'savefig.facecolor': BG,
    'axes.edgecolor': '#39456b',
    'text.color': TEXT, 'axes.labelcolor': TEXT, 'axes.titlecolor': TEXT,
    'xtick.color': DIM, 'ytick.color': DIM, 'grid.color': GRID,
    'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold',
})

COL = {'baseline': '#ef5350', 'ft': '#4fc3f7', 'distill': '#81c784', 'curriculum': '#ffb74d'}
ORDER = ['baseline', 'ft', 'distill', 'curriculum']

# ── Load data ────────────────────────────────────────────────────────────────
summary_path = sorted(glob.glob(str(ROOT / 'data/evals/summary_*.json')))[-1]
summary = json.loads(Path(summary_path).read_text())
S = summary['models']

base_rows = [json.loads(l) for l in open(ROOT / 'data/evals/baseline_scores.jsonl')]
nb = len(base_rows)
def bmean(k):
    v = [r[k] for r in base_rows if r.get(k) is not None]
    return sum(v) / len(v) if v else None

# Unified metric table: short-name -> {model: value or None}
M = {
    'baseline': {
        'json_valid': sum(1 for r in base_rows if r['json_valid']) / nb,
        'savings_valid': None, 'budget_compliance': None,
        'schema_compliance': bmean('schema_compliance'),
        'intent_alignment': None, 'rouge_l': bmean('rouge_l'),
        'bertscore_f1': bmean('bertscore_f1'), 'reasoning_coherence': None,
        'grounding_accuracy': None, 'red_team_pass': None,
    },
    'ft': S['pivotai-ft'], 'distill': S['pivotai-distill'], 'curriculum': S['pivotai-curriculum'],
}

def style_axes(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#39456b')
    ax.tick_params(length=0)

# ═════════════════════════ 1 · RADAR ═════════════════════════════════════════
def radar():
    metrics = ['json_valid', 'savings_valid', 'budget_compliance', 'schema_compliance',
               'intent_alignment', 'rouge_l', 'bertscore_f1', 'reasoning_coherence', 'grounding_accuracy']
    labels = ['JSON\nvalid', 'Savings', 'Budget', 'Schema', 'Intent\nalign',
              'ROUGE-L', 'BERTScore', 'Coherence', 'Grounding']
    N = len(metrics)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(figsize=(8.4, 8.4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)

    # draw largest-area series first so smaller ones stay visible on top
    area = {m: sum((M[m].get(k) or 0) for k in metrics) for m in ORDER}
    for name in sorted(ORDER, key=lambda m: area[m], reverse=True):
        vals = [M[name].get(k) or 0.0 for k in metrics]
        vals += vals[:1]
        c = COL[name]
        ax.plot(ang, vals, '-o', lw=2.4, ms=4.5, color=c, label=name, zorder=5)
        ax.fill(ang, vals, color=c, alpha=0.07, zorder=2)

    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, size=10, color=TEXT)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['.25', '.50', '.75', '1.0'], size=8, color=DIM)
    ax.tick_params(axis='x', pad=12)
    ax.grid(color=GRID, lw=0.7)
    ax.spines['polar'].set_color('#39456b')
    ax.set_title('pivotai — Per-Metric Comparison (4 models)', size=15, pad=28)
    leg = ax.legend(loc='upper right', bbox_to_anchor=(1.16, 1.10), frameon=True,
                    facecolor=PANEL, edgecolor='#39456b', fontsize=10, labelcolor=TEXT)
    fig.text(0.5, 0.035, 'Unmeasured metrics (e.g. baseline grounding, distill budget) shown as 0.',
             ha='center', size=8.5, color=DIM, style='italic')
    fig.savefig(CHARTS / 'radar_all_metrics.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ═════════════════════════ 2 · STRUCTURAL vs SEMANTIC ════════════════════════
def struct_sem():
    panels = [
        ('Structural', ['json_valid', 'savings_valid', 'budget_compliance', 'schema_compliance'],
         ['JSON\nvalid', 'savings\nvalid', 'budget\ncompliance', 'schema\ncompliance']),
        ('Semantic', ['rouge_l', 'bertscore_f1', 'reasoning_coherence', 'grounding_accuracy'],
         ['ROUGE-L', 'BERTScore\nF1', 'reasoning\ncoherence', 'grounding\naccuracy']),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.patch.set_facecolor(BG)
    w = 0.2
    for ax, (title, keys, labs) in zip(axes, panels):
        ax.set_facecolor(PANEL); style_axes(ax)
        x = np.arange(len(keys))
        for j, name in enumerate(ORDER):
            vals = [M[name].get(k) for k in keys]
            heights = [v if v is not None else 0.0 for v in vals]
            off = (j - 1.5) * w
            ax.bar(x + off, heights, w, color=COL[name], edgecolor=BG, linewidth=0.6,
                   label=name, zorder=3)
            for xi, v in zip(x + off, vals):
                if v is None:
                    ax.text(xi, 0.012, 'n/a', ha='center', va='bottom', size=7, color=DIM,
                            rotation=90, zorder=4)
                elif v > 0.001:
                    ax.text(xi, v + 0.015, f'{v:.2f}', ha='center', va='bottom', size=7.5,
                            color=TEXT, zorder=4)
                else:
                    ax.text(xi, 0.012, '0', ha='center', va='bottom', size=7.5, color=DIM, zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(labs, size=9.5)
        ax.set_ylim(0, 1.08); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel('Score', size=10)
        ax.set_title(title, size=13, pad=10)
        ax.grid(axis='y', alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper center', ncol=4, frameon=True, facecolor=PANEL,
               edgecolor='#39456b', fontsize=10.5, labelcolor=TEXT, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle('pivotai — Structural vs Semantic Metrics', size=16, y=1.05)
    fig.savefig(CHARTS / 'structural_vs_semantic.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ═════════════════════════ 3 · HEAD-TO-HEAD ══════════════════════════════════
def head_to_head():
    h2h = summary['head_to_head']
    fig, ax = plt.subplots(figsize=(10, 5.6)); ax.set_facecolor(PANEL); style_axes(ax)
    pairs, wa, wb, ca, cb, la, lb = [], [], [], [], [], [], []
    short = {'pivotai-ft': 'ft', 'pivotai-distill': 'distill', 'pivotai-curriculum': 'curriculum'}
    for key, v in h2h.items():
        a, b = key.split('_vs_')
        sa, sb = short[a], short[b]
        tot = v['wins_a'] + v['wins_b'] + v.get('ties', 0) or 1
        pairs.append(f'{sa}  vs  {sb}'); la.append(sa); lb.append(sb)
        wa.append(v['wins_a'] / tot * 100); wb.append(v['wins_b'] / tot * 100)
        ca.append(COL[sa]); cb.append(COL[sb])
    x = np.arange(len(pairs)); w = 0.34
    ba = ax.bar(x - w/2, wa, w, color=ca, edgecolor=BG, linewidth=0.6, zorder=3)
    bb = ax.bar(x + w/2, wb, w, color=cb, edgecolor=BG, linewidth=0.6, zorder=3)
    ax.axhline(50, color=DIM, lw=1.0, ls='--', alpha=0.6, zorder=2)
    ax.text(len(pairs)-0.5, 51.5, '50% — coin flip', ha='right', size=8.5, color=DIM, style='italic')
    for i in range(len(pairs)):
        ax.text(x[i]-w/2, wa[i]+1.2, f'{wa[i]:.0f}%', ha='center', size=9, color=TEXT, weight='bold')
        ax.text(x[i]+w/2, wb[i]+1.2, f'{wb[i]:.0f}%', ha='center', size=9, color=TEXT, weight='bold')
        ax.text(x[i]-w/2, 3, la[i], ha='center', size=8, color='#0f1626', weight='bold', rotation=90)
        ax.text(x[i]+w/2, 3, lb[i], ha='center', size=8, color='#0f1626', weight='bold', rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(pairs, size=10.5)
    ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel('Win rate (%)', size=10.5)
    ax.set_title('pivotai — Head-to-Head Win Rates (LLM judge, 92 cases)', size=14, pad=12)
    ax.grid(axis='y', alpha=0.3, zorder=0); ax.set_axisbelow(True)
    fig.savefig(CHARTS / 'head_to_head.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ═════════════════════════ 4 · RED-TEAM ══════════════════════════════════════
def red_team():
    fig, ax = plt.subplots(figsize=(10, 5.6)); ax.set_facecolor(PANEL); style_axes(ax)
    names = ORDER
    vals = [M[n].get('red_team_pass') for n in names]
    x = np.arange(len(names))
    for xi, n, v in zip(x, names, vals):
        if v is None:
            ax.bar(xi, 0.0, 0.55, color='none')
            ax.text(xi, 0.035, 'N/A', ha='center', va='bottom', size=12, color=DIM, weight='bold')
        else:
            ax.bar(xi, v, 0.55, color=COL[n], edgecolor=BG, linewidth=0.6, zorder=3)
            ax.text(xi, v + 0.012, f'{v*100:.1f}%', ha='center', va='bottom', size=11,
                    color=TEXT, weight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, size=11)
    ax.set_ylim(0, 0.8); ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.set_ylabel('Pass rate', size=10.5)
    ax.grid(axis='y', alpha=0.3, zorder=0); ax.set_axisbelow(True)
    # reserve headroom so the title and the note sit well clear of each other
    fig.subplots_adjust(top=0.80)
    fig.suptitle('pivotai — Red-Team Pass Rates (45 adversarial cases)', size=14,
                 weight='bold', y=0.965)
    fig.text(0.5, 0.875, 'Baseline not evaluated — 0% JSON validity makes structured-constraint '
             'safety metrics uninterpretable.', ha='center', size=8.8, color=DIM, style='italic')
    fig.savefig(CHARTS / 'red_team_pass.png', dpi=150, facecolor=BG)
    plt.close()

if __name__ == '__main__':
    radar(); struct_sem(); head_to_head(); red_team()
    print('Regenerated 4 charts in', CHARTS)
