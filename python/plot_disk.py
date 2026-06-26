"""
plot_disk.py  —  Black_Hole / devulkan
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser(description="Plota dados do disco de acreção")
parser.add_argument("--csv",  default="data/disk_data.csv", help="Caminho para o CSV")
parser.add_argument("--save", action="store_true",          help="Salva figuras em data/")
args = parser.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    sys.exit(f"[erro] Arquivo não encontrado: {csv_path}\n"
             f"       Rode o programa C++ com --disk primeiro.")

df = pd.read_csv(csv_path)
r  = df["r_rs"].values

print(f"[plot_disk] {len(df)} anéis carregados")
print(f"  intervalo: [{r.min():.2f}, {r.max():.2f}] RS")
print(f"  colunas:   {list(df.columns)}")

# ── Estética ──────────────────────────────────────────────────────────────────

plt.style.use("dark_background")

ORANGE = "#FF8C00"
YELLOW = "#FFD700"
BLUE   = "#4FC3F7"
RED    = "#FF5252"
GREEN  = "#69F0AE"
WHITE  = "#FFFFFF"
GRAY   = "#888888"   # mais claro que antes (#555)
BG     = "#0D0D0D"
AX_BG  = "#141414"

LABEL_KW = dict(fontsize=12, color=WHITE)
TICK_KW  = dict(labelsize=10, colors=WHITE)

def style_ax(ax, xlabel="R / RS", ylabel=""):
    """Aplica estética depois que o plot já foi desenhado."""
    ax.set_xlabel(xlabel, **LABEL_KW)
    ax.set_ylabel(ylabel, **LABEL_KW)

    # ticks brancos e maiores
    ax.tick_params(axis="both", labelsize=10, colors=WHITE,
                   length=5, width=0.8)

    # spines visíveis
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # grid mais visível
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5, color=GRAY)
    ax.set_axisbelow(True)

    # ISCO — depois do plot, ylim já está definido
    ylo, yhi = ax.get_ylim()
    ax.axvline(3.0, color=RED, linewidth=1.0, linestyle=":", alpha=0.9)
    ax.text(3.08, ylo + (yhi - ylo) * 0.88,
            "ISCO", color=RED, fontsize=9, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 1 — Emissividade e temperatura normalizada
# ─────────────────────────────────────────────────────────────────────────────

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig1.suptitle("Disco de Acreção — Perfil Radial (Schwarzschild)",
              color=WHITE, fontsize=14, fontweight="bold")
fig1.patch.set_facecolor(BG)
for ax in (ax1a, ax1b):
    ax.set_facecolor(AX_BG)

ax1a.plot(r, df["emissivity"], color=ORANGE, linewidth=2.0, label="emissividade")
ax1a.fill_between(r, df["emissivity"], alpha=0.2, color=ORANGE)
ax1a.set_ylabel("Emissividade", **LABEL_KW)
ax1a.legend(fontsize=10, framealpha=0.3, edgecolor=GRAY)
style_ax(ax1a, xlabel="")

ax1b.plot(r, df["temp_norm"], color=YELLOW, linewidth=2.0,
          label="T normalizada (Novikov-Thorne)")
ax1b.fill_between(r, df["temp_norm"], alpha=0.18, color=YELLOW)
ax1b.set_ylabel("Temperatura normalizada", **LABEL_KW)
ax1b.legend(fontsize=10, framealpha=0.3, edgecolor=GRAY)
style_ax(ax1b)

fig1.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# Figura 2 — Redshift e Doppler
# ─────────────────────────────────────────────────────────────────────────────

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig2.suptitle("Efeitos Relativísticos por Anel",
              color=WHITE, fontsize=14, fontweight="bold")
fig2.patch.set_facecolor(BG)
for ax in (ax2a, ax2b):
    ax.set_facecolor(AX_BG)

ax2a.plot(r, df["redshift"], color=RED, linewidth=2.0,
          label="redshift gravitacional  √(1 − RS/r)")
ax2a.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--")
ax2a.set_ylabel("Fator de redshift", **LABEL_KW)
ax2a.legend(fontsize=10, framealpha=0.3, edgecolor=GRAY)
style_ax(ax2a, xlabel="")

ax2b.plot(r, df["doppler_front"], color=BLUE,   linewidth=2.0,
          label="Doppler frontal  (φ=0, aproximando)")
ax2b.plot(r, df["doppler_back"],  color=ORANGE, linewidth=2.0,
          label="Doppler traseiro (φ=π, afastando)")
ax2b.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.7)
ax2b.set_ylabel("Fator Doppler  D", **LABEL_KW)
ax2b.legend(fontsize=10, framealpha=0.3, edgecolor=GRAY)
style_ax(ax2b)

fig2.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# Figura 3 — Energia observada
# ─────────────────────────────────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(10, 5))
fig3.patch.set_facecolor(BG)
ax3.set_facecolor(AX_BG)
fig3.suptitle("Energia Observada por Anel  (emissividade × D³ × redshift)",
              color=WHITE, fontsize=14, fontweight="bold")

ax3.plot(r, df["energy_obs"], color=GREEN, linewidth=2.2, label="E_obs(R)")
ax3.fill_between(r, df["energy_obs"], alpha=0.15, color=GREEN)

i_peak = int(df["energy_obs"].idxmax())
r_peak = r[i_peak]
e_peak = float(df["energy_obs"].iloc[i_peak])
ax3.axvline(r_peak, color=YELLOW, linewidth=1.2, linestyle="--", alpha=0.9)

style_ax(ax3, ylabel="Energia observada (u.a.)")

# texto do pico depois de style_ax (ylim já definido)
ylo3, yhi3 = ax3.get_ylim()
ax3.text(r_peak + 0.15, ylo3 + (yhi3 - ylo3) * 0.75,
         f"pico em R = {r_peak:.2f} RS", color=YELLOW, fontsize=10)

ax3.legend(fontsize=10, framealpha=0.3, edgecolor=GRAY)
fig3.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# Figura 4 — Luminosidade por anel
# ─────────────────────────────────────────────────────────────────────────────

fig4, ax4 = plt.subplots(figsize=(10, 5))
fig4.patch.set_facecolor(BG)
ax4.set_facecolor(AX_BG)
fig4.suptitle("Luminosidade por Anel  L(R) = emissividade × 2πR ΔR",
              color=WHITE, fontsize=14, fontweight="bold")

ax4.plot(r, df["luminosity"], color=ORANGE, linewidth=2.2, label="L(R)")
ax4.fill_between(r, df["luminosity"], alpha=0.18, color=ORANGE)

style_ax(ax4, ylabel="Luminosidade (u.a.)")
ax4.legend(fontsize=10, framealpha=0.3, edgecolor=GRAY)
fig4.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────

if args.save:
    out = Path("data")
    fig1.savefig(out / "disk_emissivity_temp.png", dpi=150,
                 bbox_inches="tight", facecolor=fig1.get_facecolor())
    fig2.savefig(out / "disk_relativistic.png",    dpi=150,
                 bbox_inches="tight", facecolor=fig2.get_facecolor())
    fig3.savefig(out / "disk_energy_obs.png",      dpi=150,
                 bbox_inches="tight", facecolor=fig3.get_facecolor())
    fig4.savefig(out / "disk_luminosity.png",      dpi=150,
                 bbox_inches="tight", facecolor=fig4.get_facecolor())
    print("[plot_disk] Figuras salvas em data/")
else:
    plt.show()