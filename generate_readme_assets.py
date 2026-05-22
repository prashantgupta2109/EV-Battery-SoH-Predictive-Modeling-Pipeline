# =============================================================================
#  generate_readme_assets.py
#  Generates all charts and saves them as PNG files for the README.
#  Run once: python generate_readme_assets.py
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data_battery_cycle")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
ASSETS_DIR = os.path.join(BASE_DIR, "readme_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

FEATURES = ["voltage", "temperature", "cycle"]
TARGET   = "soh"

# ---------------------------------------------------------------------------
# Shared dark theme
# ---------------------------------------------------------------------------
BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#30363d"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
ORANGE  = "#f0883e"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"

def apply_dark(fig, axes=None):
    fig.patch.set_facecolor(BG)
    if axes is None:
        return
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        if ax.get_title():
            ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

# ---------------------------------------------------------------------------
# Load data & model
# ---------------------------------------------------------------------------
import glob
csv = glob.glob(os.path.join(DATA_DIR, "*.csv"))[0]
df  = pd.read_csv(csv)
df.columns = [c.lower().strip() for c in df.columns]
df[TARGET]  = df[TARGET].clip(0.0, 1.2)

model        = joblib.load(os.path.join(MODEL_DIR, "battery_model.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names= joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
metrics      = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))

X = df[FEATURES]; y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=FEATURES)
y_pred     = model.predict(X_test_sc)

PALETTE = [BLUE, GREEN, ORANGE, RED, "#d2a8ff", "#ffa657", "#79c0ff", "#56d364"]

# ============================================================
# PLOT 1 — SoH degradation curves per battery
# ============================================================
print("[1/6] SoH Degradation Curves ...")
batteries = sorted(df["battery_id"].unique()) if "battery_id" in df.columns else []

fig, ax = plt.subplots(figsize=(11, 5))
apply_dark(fig, ax)
ax.axhline(0.80, color=RED, linewidth=1.6, linestyle="--", label="80% End-of-Life")
ax.fill_between([0, df["cycle"].max()], 0, 0.80,
                alpha=0.12, color=RED, zorder=0)

for i, bid in enumerate(batteries):
    sub = df[df["battery_id"] == bid].sort_values("cycle")
    smooth = sub["soh"].rolling(5, min_periods=1).mean()
    ax.plot(sub["cycle"], smooth, linewidth=2,
            color=PALETTE[i % len(PALETTE)], label=bid, alpha=0.9)

ax.set_xlabel("Charge Cycle Number", color=TEXT, fontsize=11)
ax.set_ylabel("State of Health (SoH)", color=TEXT, fontsize=11)
ax.set_title("Battery SoH Degradation Over Charge Cycles", color=TEXT, fontsize=13, fontweight="bold", pad=14)
ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax.set_ylim(0.55, 1.08)
ax.grid(axis="y", color=BORDER, linewidth=0.6, linestyle=":")
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "plot1_soh_degradation.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 2 — Capacity fade per battery
# ============================================================
print("[2/6] Capacity Fade Curves ...")
fig, ax = plt.subplots(figsize=(11, 5))
apply_dark(fig, ax)
for i, bid in enumerate(batteries):
    sub = df[df["battery_id"] == bid].sort_values("cycle")
    smooth = sub["capacity"].rolling(5, min_periods=1).mean()
    ax.plot(sub["cycle"], smooth, linewidth=2,
            color=PALETTE[i % len(PALETTE)], label=bid, alpha=0.9)
    ax.fill_between(sub["cycle"], smooth, smooth.min(),
                    alpha=0.06, color=PALETTE[i % len(PALETTE)])

ax.set_xlabel("Charge Cycle Number", color=TEXT, fontsize=11)
ax.set_ylabel("Discharge Capacity (Ah)", color=TEXT, fontsize=11)
ax.set_title("Discharge Capacity Fade Across Charge Cycles", color=TEXT, fontsize=13, fontweight="bold", pad=14)
ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax.grid(axis="y", color=BORDER, linewidth=0.6, linestyle=":")
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "plot2_capacity_fade.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 3 — Temperature vs Cycle (scatter coloured by SoH)
# ============================================================
print("[3/6] Temperature vs Cycle Scatter ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
apply_dark(fig, axes)

sc = axes[0].scatter(df["cycle"], df["temperature"],
                     c=df["soh"], cmap="RdYlGn",
                     s=12, alpha=0.65, vmin=0.6, vmax=1.0)
cbar = fig.colorbar(sc, ax=axes[0], pad=0.02)
cbar.set_label("State of Health", color=TEXT, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
cbar.outline.set_edgecolor(BORDER)
axes[0].set_xlabel("Charge Cycle Number", color=TEXT, fontsize=10)
axes[0].set_ylabel("Temperature (°C)", color=TEXT, fontsize=10)
axes[0].set_title("Temperature vs Cycle\n(colour = SoH)", color=TEXT, fontsize=11, fontweight="bold")
axes[0].grid(color=BORDER, linewidth=0.5, linestyle=":")

# Temperature histogram
axes[1].hist(df["temperature"], bins=50, color=BLUE, alpha=0.8, edgecolor=BG, linewidth=0.4)
axes[1].set_xlabel("Temperature (°C)", color=TEXT, fontsize=10)
axes[1].set_ylabel("Count", color=TEXT, fontsize=10)
axes[1].set_title("Temperature Distribution\nAcross All Cycles", color=TEXT, fontsize=11, fontweight="bold")
axes[1].grid(axis="y", color=BORDER, linewidth=0.5, linestyle=":")

plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "plot3_temperature.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 4 — Actual vs Predicted SoH
# ============================================================
print("[4/6] Actual vs Predicted ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
apply_dark(fig, axes)

# Scatter
lo, hi = min(y_test.min(), y_pred.min()) - 0.01, max(y_test.max(), y_pred.max()) + 0.01
axes[0].scatter(y_test, y_pred, s=18, alpha=0.55, color=BLUE, edgecolors="none")
axes[0].plot([lo, hi], [lo, hi], color=GREEN, linewidth=2, linestyle="--", label="Perfect fit")
axes[0].set_xlabel("Actual SoH", color=TEXT, fontsize=10)
axes[0].set_ylabel("Predicted SoH", color=TEXT, fontsize=10)
axes[0].set_title("Actual vs Predicted SoH\n(Test Set)", color=TEXT, fontsize=11, fontweight="bold")
axes[0].legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
axes[0].grid(color=BORDER, linewidth=0.5, linestyle=":")

# Residuals histogram
residuals = y_pred - y_test.values
axes[1].hist(residuals, bins=40, color=ORANGE, alpha=0.85, edgecolor=BG, linewidth=0.4)
axes[1].axvline(0, color=GREEN, linewidth=1.8, linestyle="--", label="Zero error")
axes[1].axvline(residuals.mean(), color=RED, linewidth=1.6, linestyle=":", label=f"Mean={residuals.mean():.4f}")
axes[1].set_xlabel("Residual (Predicted - Actual)", color=TEXT, fontsize=10)
axes[1].set_ylabel("Count", color=TEXT, fontsize=10)
axes[1].set_title("Residual Distribution\n(Test Set)", color=TEXT, fontsize=11, fontweight="bold")
axes[1].legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
axes[1].grid(axis="y", color=BORDER, linewidth=0.5, linestyle=":")

plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "plot4_actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 5 — Feature Importance
# ============================================================
print("[5/6] Feature Importance ...")
importances = model.feature_importances_
feat_names  = model.feature_name_

fig, ax = plt.subplots(figsize=(8, 4))
apply_dark(fig, ax)
colors = [BLUE, GREEN, ORANGE]
bars = ax.barh(feat_names, importances, color=colors, edgecolor=BG, height=0.55)
for bar, val in zip(bars, importances):
    ax.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2,
            f"{val:,.0f}", va="center", color=TEXT, fontsize=10, fontweight="bold")
ax.set_xlabel("Feature Importance Score (split gain)", color=TEXT, fontsize=10)
ax.set_title("LightGBM Feature Importances", color=TEXT, fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="x", color=BORDER, linewidth=0.6, linestyle=":")
ax.set_xlim(0, max(importances) * 1.18)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "plot5_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 6 — Metrics dashboard card
# ============================================================
print("[6/6] Metrics Summary Card ...")
r2   = metrics["r2"]
rmse = metrics["rmse"]
mae  = np.mean(np.abs(y_pred - y_test.values))
mape = np.mean(np.abs((y_pred - y_test.values) / y_test.values)) * 100
max_err = np.max(np.abs(y_pred - y_test.values))

fig = plt.figure(figsize=(12, 3.5))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

metric_data = [
    ("R² Score",   f"{r2*100:.2f}%",    GREEN,  "Variance explained\nby the model"),
    ("RMSE",       f"{rmse:.5f}",        BLUE,   "Root Mean\nSquared Error"),
    ("MAE",        f"{mae:.5f}",         ORANGE, "Mean Absolute\nError"),
    ("MAPE",       f"{mape:.3f}%",       "#d2a8ff", "Mean Absolute\nPercentage Error"),
    ("Max Error",  f"{max_err:.5f}",     RED,    "Worst-case\nresidual"),
]
for idx, (label, value, color, desc) in enumerate(metric_data):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.72, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=20, fontweight="bold", color=color)
    ax.text(0.5, 0.38, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color=TEXT, fontweight="600")
    ax.text(0.5, 0.12, desc, transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, color=MUTED, linespacing=1.4)

plt.suptitle("Model Evaluation Metrics  —  LightGBM Regressor  (Test Set, 283 samples)",
             color=TEXT, fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "plot6_metrics_card.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n[DONE] All 6 assets saved to:", ASSETS_DIR)
for f in sorted(os.listdir(ASSETS_DIR)):
    print(f"  {f}")
