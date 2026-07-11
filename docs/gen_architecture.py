"""pivotai architecture diagram — dark theme matching the Phase 4 eval charts.

Layered systems diagram: data generation, the agent -> MCP -> external-API call
stack, QLoRA training, evaluation, and FastAPI serving.

Layout is computed top-to-bottom with a running cursor so every phase card is
sized to contain its contents (no text overflow, no elements outside cards).
Pure matplotlib, headless."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Arc, RegularPolygon
from pathlib import Path
import numpy as np

# ── Palette (identical to results_analysis.ipynb) ────────────────────────────
BG, CARD, INNER, INNER2 = '#1a1a2e', '#16213e', '#1f2a48', '#24345c'
TEXT, DIM = '#e6e8ef', '#98a1ba'
C1, C2, C3, C4, C5 = '#fb923c', '#4ade80', '#c084fc', '#f87171', '#38bdf8'
GREY = '#7b88a1'
M_FT, M_DIST, M_CURR, M_BASE = '#4fc3f7', '#81c784', '#ffb74d', '#ef5350'
ARROW, THIN = '#cdd3e0', '#7c8aa5'
PILL, PILLE = '#243054', '#46568a'

X0, WROOT = 0.3, 14.4          # phase card x and width
XIN_L, XIN_R = X0 + 0.3, X0 + WROOT - 0.3
XC = X0 + WROOT / 2            # horizontal centre
HB = 0.64                      # header band height

# ── Canvas ───────────────────────────────────────────────────────────────────
FW, FH = 15.0, 28.4
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.set_aspect('equal'); ax.axis('off')

# ── Primitives ───────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.5, r=0.15, z=4, ls='solid'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                 facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))

def txt(x, y, s, size=10, color=TEXT, weight='normal', ha='center', va='center', z=7, style='normal'):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            zorder=z, fontfamily='DejaVu Sans', fontstyle=style)

def phase_frame(yb, yt, color, num, title, sub=''):
    """Rounded phase card drawn BEHIND internals (low zorder)."""
    rbox(X0, yb, WROOT, yt - yb, CARD, color, lw=2.0, r=0.26, z=2)
    rbox(X0, yt - HB, WROOT, HB, color, color, lw=0, r=0.26, z=3)
    ax.add_patch(Rectangle((X0, yt - HB), WROOT, HB - 0.26, facecolor=color, edgecolor='none', zorder=3))
    ax.plot([X0, X0 + WROOT], [yt - HB, yt - HB], color=BG, lw=1.2, zorder=3.5)
    cy = yt - HB / 2
    ax.add_patch(plt.Circle((X0 + 0.52, cy), 0.235, color='white', zorder=5))
    txt(X0 + 0.52, cy, str(num), size=14, color=color, weight='bold', z=6)
    txt(X0 + 0.95, cy, title, size=12, color='#0f1626', weight='bold', ha='left', z=6)
    if sub:
        txt(X0 + WROOT - 0.3, cy, sub, size=9, color='#0f1626', weight='bold', ha='right', z=6)

def node(x, y, w, h, ec, title, lines=(), fc=INNER, tcolor=None, lw=1.6, z=5, tsize=10.5):
    """Box with a bold title and dim sub-lines, vertically centred in the box."""
    rbox(x, y, w, h, fc, ec, lw=lw, r=0.15, z=z)
    n = len(lines)
    span = 0.44 + (n - 1) * 0.31 if n else 0.0      # title-to-last-line span
    ty = y + h / 2 + span / 2
    txt(x + w / 2, ty, title, size=tsize, color=tcolor or ec, weight='bold', z=z + 1)
    for i, ln in enumerate(lines):
        txt(x + w / 2, ty - 0.44 - i * 0.31, ln, size=8.6, color=DIM, z=z + 1)

def cylinder(cx, by, w, h, ec, label, sub=''):
    eh = min(0.32, h * 0.28)                          # shallower lid on short cylinders
    bcy, tcy = by + eh / 2, by + h - eh / 2
    ax.add_patch(Rectangle((cx - w / 2, bcy), w, h - eh, facecolor=INNER, edgecolor='none', zorder=5))
    ax.plot([cx - w / 2, cx - w / 2], [bcy, tcy], color=ec, lw=1.6, zorder=6)
    ax.plot([cx + w / 2, cx + w / 2], [bcy, tcy], color=ec, lw=1.6, zorder=6)
    ax.add_patch(Arc((cx, bcy), w, eh, theta1=180, theta2=360, edgecolor=ec, lw=1.6, zorder=6))
    ax.add_patch(Ellipse((cx, tcy), w, eh, facecolor=INNER2, edgecolor=ec, lw=1.6, zorder=7))
    # label/sub centred in the visible front face, clear of both ellipse rims
    if sub:
        txt(cx, by + h * 0.50, label, size=9.5, color=ec, weight='bold', z=8)
        txt(cx, by + h * 0.26, sub, size=8.0, color=DIM, z=8)
    else:
        txt(cx, by + (bcy + tcy) / 2 - by, label, size=9.5, color=ec, weight='bold', z=8)

def hexagon(cx, cy, r, ec, label, sub=''):
    ax.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=r, orientation=np.pi / 6,
                 facecolor=INNER, edgecolor=ec, lw=1.6, zorder=5))
    txt(cx, cy + 0.11, label, size=8.7, color=ec, weight='bold', z=7)
    if sub:
        txt(cx, cy - 0.17, sub, size=7.6, color=DIM, z=7)

def arr_down(xc, y_top, y_bot, color=THIN, lw=1.7, z=4):
    ax.annotate('', xy=(xc, y_bot), xytext=(xc, y_top),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=15), zorder=z)

def arr_h(x1, x2, y, color=THIN, lw=1.7, z=4):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=15), zorder=z)

def big_arrow(y_top, y_bot, label):
    ax.annotate('', xy=(XC, y_bot), xytext=(XC, y_top),
                arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=3.8, mutation_scale=28), zorder=4)
    pw, ph = 0.158 * len(label) + 0.7, 0.48
    mx, my = XC + 0.32 + pw / 2, (y_top + y_bot) / 2
    rbox(mx - pw / 2, my - ph / 2, pw, ph, PILL, PILLE, lw=1.2, r=0.18, z=7)
    txt(mx, my, label, size=9.4, color=TEXT, weight='bold', z=8)

def loop(y1, y2, label):
    """Double-headed vertical arrow (request/response) with a side pill."""
    ax.annotate('', xy=(XC, y2), xytext=(XC, y1),
                arrowprops=dict(arrowstyle='<|-|>', color=C2, lw=2.4, mutation_scale=16), zorder=4)
    pw = 0.150 * len(label) + 0.55
    rbox(XC + 0.4, (y1 + y2) / 2 - 0.22, pw, 0.44, '#1c3326', '#2f6b44', lw=1.1, r=0.16, z=7)
    txt(XC + 0.4 + pw / 2, (y1 + y2) / 2, label, size=8.5, color='#bbf7d0', weight='bold', z=8)

def chip(cx, cy, w, text, color):
    rbox(cx - w / 2, cy - 0.18, w, 0.36, '#0f1a30', color, lw=1.1, r=0.15, z=6)
    txt(cx, cy, text, size=7.9, color=color, weight='bold', z=7)

# ═════════════════════════════════════════════════════════════════════════════
# Layout — running cursor from the top
# ═════════════════════════════════════════════════════════════════════════════
cur = FH - 0.55

# ── Title ────────────────────────────────────────────────────────────────────
txt(XC, cur, 'pivotai — System Architecture', size=18.5, color=TEXT, weight='bold')
txt(XC, cur - 0.5, 'multi-agent travel optimizer  ·  data → agents → training → evaluation → serving',
    size=10, color=DIM, style='italic')
ax.plot([XC - 2.5, XC + 2.5], [cur - 0.85, cur - 0.85], color=C5, lw=2.4, zorder=4)
cur -= 1.35                                    # top available for phase 1

PAD = 0.30          # padding under header / above card bottom
ARRH = 0.95         # inter-phase arrow length
GAP = 0.15          # gap after arrow before next phase

# ════════════════════════ PHASE 1 — DATA GENERATION ══════════════════════════
top = cur
ct = top - HB - PAD
bh = 1.45
by = ct - bh
node(0.6, by, 3.6, bh, C1, 'GPT-4o-mini teacher',
     ('5,000 prompts · 20 cities', '5 budgets · 8 intents'))
arr_h(4.2, 4.75, by + bh / 2)
node(4.75, by, 4.05, bh, C1, '3-Gate Validator',
     ('hostel · savings ≥ 5%', 'budget · ~12% rejected'))
arr_h(8.8, 9.35, by + bh / 2)
cylinder(11.85, by, 4.5, bh, C1, '5,000 training pairs', '(baseline, optimized) + pivot')
bot = by - PAD
phase_frame(bot, top, C1, 1, 'Synthetic Data Engine', 'OpenAI gpt-4o-mini · $4')
big_arrow(bot, bot - ARRH, 'persona → itinerary pairs')
cur = bot - ARRH - GAP

# ════════════════════════ PHASE 2 — AGENTIC PIPELINE ═════════════════════════
top = cur
c = top - HB - PAD                              # internal cursor
# Supervisor
sup_h = 0.86
sy = c - sup_h
rbox(3.7, sy, 7.6, sup_h, INNER2, C2, lw=1.8, r=0.16, z=5)
txt(5.0, sy + sup_h / 2, 'Supervisor', size=11, color=C2, weight='bold', z=6)
txt(8.7, sy + sup_h - 0.30, 'async · concurrency = 3 · checkpoint-resume', size=8.4, color=DIM, z=6)
txt(8.7, sy + 0.28, 'quality filter → 500 clean traces (from 545)', size=8.4, color=DIM, z=6)
arr_down(XC, sy, sy - 0.34)
# Agent chain
ag_h = 1.5
ag_y = sy - 0.40 - ag_h
aw, axs = 4.3, [0.7, 5.3, 9.9]
for (name, role, out), bx in zip(
        [('Analyst', 'identifies cost drivers', '→ cost_report'),
         ('Concierge', 'finds cheaper swaps', '→ substitutions'),
         ('Optimizer', 'builds final itinerary', '→ pivot_analysis')], axs):
    rbox(bx, ag_y, aw, ag_h, INNER, C2, lw=1.6, r=0.16, z=5)
    txt(bx + aw / 2, ag_y + ag_h - 0.36, name, size=11.5, color=C2, weight='bold', z=6)
    txt(bx + aw / 2, ag_y + ag_h - 0.76, role, size=8.7, color=TEXT, z=6)
    txt(bx + aw / 2, ag_y + 0.32, out, size=9.4, color=C2, weight='bold', z=6)
arr_h(axs[0] + aw, axs[1], ag_y + ag_h / 2, color=C2, lw=2.0)
arr_h(axs[1] + aw, axs[2], ag_y + ag_h / 2, color=C2, lw=2.0)
# loop: agents <-> MCP
loop(ag_y, ag_y - 0.88, 'function call  ↕  tool result')
# MCP layer
txt(XC, ag_y - 1.10, 'MCP TOOL LAYER  ·  official mcp library  ·  SSE transport  ·  @api_cache',
    size=8.6, color=C2, weight='bold', z=6)
mcp_h = 1.0
mcp_y = ag_y - 1.35 - mcp_h
mw, mxs = 3.25, [0.7, 4.3, 7.9, 11.5]
for (n, t), bx in zip(
        [('routing :8001', 'get_route'), ('hotels :8002', 'search_hotels'),
         ('overpass :8003', 'search_pois'), ('search :8004', 'web_search')], mxs):
    node(bx, mcp_y, mw, mcp_h, C2, n, (t,), tsize=9.4)
# loop: MCP <-> external APIs
loop(mcp_y, mcp_y - 0.82, 'HTTP  ↕  JSON  (cached)')
# External API hexagons
txt(XC, mcp_y - 1.02, 'EXTERNAL DATA SOURCES  ·  free APIs', size=8.6, color=DIM, weight='bold', z=6)
hex_r = 0.62
hex_cy = mcp_y - 1.28 - hex_r
for (n, s), bx in zip(
        [('ORS', 'routing'), ('Overpass', 'hotels'), ('Overpass', 'POIs'), ('DuckDuckGo', 'search')], mxs):
    hexagon(bx + mw / 2, hex_cy, hex_r, GREY, n, s)
bot = hex_cy - hex_r - PAD
phase_frame(bot, top, C2, 2, 'Multi-Agent Orchestration', 'DeepSeek V4 Flash · $4')
big_arrow(bot, bot - ARRH, '500 reasoning traces')
cur = bot - ARRH - GAP

# ════════════════════════ PHASE 3 — TRAINING ═════════════════════════════════
top = cur
c = top - HB - PAD
card_h = 1.95
cy = c - card_h
# data lake (left)
txt(2.0, cy + card_h - 0.12, 'TRAINING DATA', size=8.2, color=DIM, weight='bold', z=6)
cylinder(2.0, cy + 0.95, 2.3, 0.78, C1, '5,000 pairs', 'Phase 1')
cylinder(2.0, cy + 0.06, 2.3, 0.78, C2, '500 traces', 'Phase 2')
arr_h(3.3, 3.85, cy + card_h / 2, color=ARROW, lw=2.2)
# three model cards
cw, cxs = 3.3, [3.9, 7.4, 10.9]
cards = [
    (M_FT,   'pivotai-ft',         'trained on 5,000 pairs',  ('SFT · Colab T4 · fp16', '3 epochs · loss 0.225')),
    (M_DIST, 'pivotai-distill',    'trained on 500 traces',   ('KD · A100 · bf16', '5 epochs · loss 0.254')),
    (M_CURR, 'pivotai-curriculum', '5,000 pairs → 500 traces', ('2-stage · lr decay 4×', 'loss 0.241 / 0.505')),
]
for (col, name, data, lines), bx in zip(cards, cxs):
    rbox(bx, cy, cw, card_h, INNER, col, lw=1.7, r=0.16, z=5)
    txt(bx + cw / 2, cy + card_h - 0.32, name, size=10.5, color=col, weight='bold', z=6)
    chip(bx + cw / 2, cy + card_h - 0.78, cw - 0.5, data, col)
    txt(bx + cw / 2, cy + 0.78, lines[0], size=8.3, color=DIM, z=6)
    txt(bx + cw / 2, cy + 0.50, lines[1], size=8.3, color=DIM, z=6)
    txt(bx + cw / 2, cy + 0.22, '4.6 GB GGUF Q4_K_M', size=8.0, color=DIM, weight='bold', z=6)
bot = cy - PAD
phase_frame(bot, top, C3, 3, 'QLoRA Fine-Tuning', 'Llama 3.1 8B · Unsloth · r=8')
big_arrow(bot, bot - ARRH, '3 fine-tuned models  +  baseline')
cur = bot - ARRH - GAP

# ════════════════════════ PHASE 4 — EVALUATION ═══════════════════════════════
top = cur
c = top - HB - PAD
eh = 1.9
ey = c - eh
# model list (left)
rbox(0.6, ey, 2.25, eh, INNER, C4, lw=1.6, r=0.15, z=5)
txt(1.725, ey + eh - 0.34, '4 models', size=9.6, color=C4, weight='bold', z=6)
for i, (m, col) in enumerate([('baseline', M_BASE), ('ft', M_FT), ('distill', M_DIST), ('curriculum', M_CURR)]):
    txt(1.725, ey + eh - 0.74 - i * 0.30, m, size=8.5, color=col, weight='bold', z=6)
arr_h(2.85, 3.4, ey + eh / 2)
ew, exs = 2.8, [3.4, 6.38, 9.36]
for (name, lines), bx in zip(
        [('Automated Metrics', ('JSON · savings · budget', 'ROUGE-L · BERTScore', 'all-MiniLM-L6-v2')),
         ('LLM-as-Judge', ('DeepSeek V4 Flash', 'reasoning coherence', 'grounding accuracy')),
         ('Red Teaming · 45', ('adversarial prompts', 'budget overrides', 'injection attempts'))], exs):
    node(bx, ey, ew, eh, C4, name, lines, tsize=10)
arr_h(exs[-1] + ew, 12.7, ey + eh / 2)
cylinder(13.55, ey, 1.85, eh, C4, 'results', 'summary + charts')
bot = ey - PAD
phase_frame(bot, top, C4, 4, 'Evaluation & Red Teaming', '92 cases · 10 metrics')
big_arrow(bot, bot - ARRH, 'eval summary  +  4 charts')
cur = bot - ARRH - GAP

# ════════════════════════ PHASE 5 — SERVING ══════════════════════════════════
top = cur
c = top - HB - PAD
nh = 1.4
ny = c - nh
sw, sxs = 3.25, [0.6, 4.07, 7.54, 11.01]
node(sxs[0], ny, sw, nh, C5, 'FastAPI app', ('async httpx', 'Pydantic validation'))
arr_h(sxs[0] + sw, sxs[1], ny + nh / 2, color=C5, lw=2.0)
node(sxs[1], ny, sw, nh, C5, 'Ollama runtime', ('local CPU', 'no GPU needed'))
arr_h(sxs[1] + sw, sxs[2], ny + nh / 2, color=C5, lw=2.0)
node(sxs[2], ny, sw, nh, C5, '4 GGUF models', ('Q4_K_M · 4.6 GB', 'registry-validated'))
node(sxs[3], ny, sw, nh, C5, 'Swagger UI  /docs', ('auto-generated', 'OpenAPI schema'),
     fc=INNER2, tcolor='#7dd3fc', lw=2.2)
# endpoint chips
eps = [('GET /health', C5), ('GET /models', C5), ('POST /optimize', '#7dd3fc'),
       ('GET /results/summary', C5), ('GET /results/compare', C5)]
widths = [0.135 * len(n) + 0.55 for n, _ in eps]
total = sum(widths) + 0.3 * (len(eps) - 1)
ex = XC - total / 2
chip_y = ny - 0.42
for (name, col), w in zip(eps, widths):
    chip(ex + w / 2, chip_y, w, name, col)
    ex += w + 0.3
bot = chip_y - 0.36
phase_frame(bot, top, C5, 5, 'FastAPI Inference Server', 'async · Pydantic · /docs')

# ── Save ─────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / 'architecture.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG, pad_inches=0.3)
plt.close()
print(f'Saved: {out}  (content bottom y = {bot:.2f})')
