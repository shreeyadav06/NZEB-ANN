"""
Publication-Grade Figures for NZEB Cross-Climate Research Paper
Exact IEEE / Interspeech publication aesthetic matching technical_report.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Standard publication font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

def draw_ann_architecture(output_path):
    """
    Renders single-column ANN Architecture figure (Fig. 2 in paper).
    Aspect ratio suited for IEEE column width (approx 3.5 inches).
    """
    fig, ax = plt.subplots(figsize=(6.4, 9.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Palette - Academic Muted Pastels (matching technical_report.pdf)
    c_blue_bg = '#F2F6FA'
    c_blue_border = '#688FAE'
    c_peach_bg = '#FDF6F0'
    c_peach_border = '#D6896C'
    c_mint_bg = '#EFF7F1'
    c_mint_border = '#559E68'
    c_gray_bg = '#F8F9FA'
    c_gray_border = '#A8B3BC'
    c_box_bg = '#FFFFFF'
    c_text_dark = '#1A2530'
    c_text_muted = '#4A5568'
    c_arrow = '#2C3E50'

    # --- 1. INPUT BLOCK (y: 91.0 - 98.5) ---
    rect_input = patches.FancyBboxPatch(
        (6, 91.0), 88, 7.8,
        boxstyle="round,pad=0.35,rounding_size=1.0",
        linewidth=1.0, edgecolor=c_gray_border, facecolor=c_gray_bg
    )
    ax.add_patch(rect_input)
    ax.text(37, 95.8, r"Input Feature Vector: $\mathbf{x} \in \mathbb{R}^{16}$", 
            ha='center', va='center', fontsize=9.8, fontweight='bold', color=c_text_dark)
    ax.text(37, 92.8, r"Weather (4: $T_{\mathrm{air}}, T_{\mathrm{dew}}, v, P$) $\cdot$ Cyclic Time (4: $\sin/\cos$) $\cdot$ Typology (8)",
            ha='center', va='center', fontsize=7.6, color=c_text_muted)

    # Mini sparkline inside Input block (cyclic sine wave)
    sub_ax_in = fig.add_axes([0.76, 0.908, 0.14, 0.032])
    t = np.linspace(0, 2*np.pi, 60)
    sub_ax_in.plot(t, np.sin(t), color='#2B6CB0', lw=1.1)
    sub_ax_in.plot(t, np.cos(t), color='#C53030', lw=1.1, ls='--')
    sub_ax_in.axis('off')

    # Arrow: Input -> Frozen Block
    ax.annotate("", xy=(50, 85.5), xytext=(50, 91.0),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=5", lw=1.0, color=c_arrow))
    ax.text(53, 88.2, r"$\mathbf{x} \in \mathbb{R}^{16}$", fontsize=7.8, va='center', color=c_arrow)

    # --- 2. FROZEN CONTAINER (Layers 1-3, y: 43.5 - 85.5) ---
    rect_frozen = patches.FancyBboxPatch(
        (4, 43.5), 92, 42.0,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        linewidth=1.1, edgecolor=c_blue_border, facecolor=c_blue_bg, linestyle='--'
    )
    ax.add_patch(rect_frozen)

    # Container Header
    ax.text(8, 83.2, "Generalized Physics Layers (Frozen in Transfer)", 
            fontsize=9.2, fontweight='bold', color='#1B365D')
    ax.text(92, 83.2, r"$\theta_{\mathrm{frozen}} =$ 43,584 params", 
            fontsize=8.5, fontweight='bold', ha='right', color='#1B365D')
    ax.text(8, 80.8, "Base weights locked from source domain (Zone 2B Tempe); preserves thermodynamic invariants",
            fontsize=7.4, style='italic', color=c_text_muted)

    # Layer 1 Card (y: 69.5 - 78.5)
    r_l1 = patches.FancyBboxPatch(
        (7, 69.5), 86, 9.2,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.9, edgecolor=c_blue_border, facecolor=c_box_bg
    )
    ax.add_patch(r_l1)
    ax.text(10, 75.8, "Dense Layer 1: 256 Units", fontsize=8.8, fontweight='bold', color=c_text_dark)
    ax.text(10, 73.2, r"$\mathbf{h}_1 = \mathrm{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$ $\cdot$ Captures cross-feature non-linearities", 
            fontsize=7.5, color=c_text_muted)
    ax.text(10, 71.0, r"Output dimension: $\mathbf{h}_1 \in \mathbb{R}^{256}$", fontsize=7.2, color='#4A5568')
    ax.text(90, 75.8, "4,352 weights", ha='right', fontsize=8.0, fontweight='bold', color='#2B6CB0')
    ax.text(90, 72.4, r"$\mathbf{W}_1 \in \mathbb{R}^{256 \times 16},\ \mathbf{b}_1 \in \mathbb{R}^{256}$", ha='right', fontsize=7.2, color=c_text_muted)

    # Arrow L1 -> L2
    ax.annotate("", xy=(50, 65.0), xytext=(50, 69.5),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=4.5", lw=0.9, color=c_arrow))
    ax.text(53, 67.2, r"$\mathbf{h}_1 \in \mathbb{R}^{256}$", fontsize=7.4, va='center', color=c_arrow)

    # Layer 2 Card (y: 56.0 - 65.0)
    r_l2 = patches.FancyBboxPatch(
        (7, 56.0), 86, 9.0,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.9, edgecolor=c_blue_border, facecolor=c_box_bg
    )
    ax.add_patch(r_l2)
    ax.text(10, 62.3, "Dense Layer 2: 128 Units", fontsize=8.8, fontweight='bold', color=c_text_dark)
    ax.text(10, 59.7, r"$\mathbf{h}_2 = \mathrm{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$ $\cdot$ Captures diurnal occupancy & solar phase", 
            fontsize=7.5, color=c_text_muted)
    ax.text(10, 57.5, r"Output dimension: $\mathbf{h}_2 \in \mathbb{R}^{128}$", fontsize=7.2, color='#4A5568')
    ax.text(90, 62.3, "32,896 weights", ha='right', fontsize=8.0, fontweight='bold', color='#2B6CB0')
    ax.text(90, 59.0, r"$\mathbf{W}_2 \in \mathbb{R}^{128 \times 256},\ \mathbf{b}_2 \in \mathbb{R}^{128}$", ha='right', fontsize=7.2, color=c_text_muted)

    # Arrow L2 -> L3
    ax.annotate("", xy=(50, 51.5), xytext=(50, 56.0),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=4.5", lw=0.9, color=c_arrow))
    ax.text(53, 53.7, r"$\mathbf{h}_2 \in \mathbb{R}^{128}$", fontsize=7.4, va='center', color=c_arrow)

    # Layer 3 Card (y: 44.2 - 51.5)
    r_l3 = patches.FancyBboxPatch(
        (7, 44.2), 86, 7.3,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.9, edgecolor=c_blue_border, facecolor=c_box_bg
    )
    ax.add_patch(r_l3)
    ax.text(10, 49.3, "Dense Layer 3: 64 Units", fontsize=8.8, fontweight='bold', color=c_text_dark)
    ax.text(10, 47.0, r"$\mathbf{h}_3 = \mathrm{ReLU}(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3)$ $\cdot$ Encodes structural thermal capacitance", 
            fontsize=7.5, color=c_text_muted)
    ax.text(10, 45.2, r"Output dimension: $\mathbf{h}_3 \in \mathbb{R}^{64}$ (bottleneck physics embedding)", fontsize=7.2, color='#4A5568')
    ax.text(90, 49.3, "8,256 weights", ha='right', fontsize=8.0, fontweight='bold', color='#2B6CB0')
    ax.text(90, 46.5, r"$\mathbf{W}_3 \in \mathbb{R}^{64 \times 128},\ \mathbf{b}_3 \in \mathbb{R}^{64}$", ha='right', fontsize=7.2, color=c_text_muted)

    # Arrow Frozen -> Trainable Block
    ax.annotate("", xy=(50, 36.5), xytext=(50, 43.5),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=5", lw=1.0, color=c_arrow))
    ax.text(53, 40.0, r"$\mathbf{h}_3 \in \mathbb{R}^{64}$", fontsize=7.8, va='center', color=c_arrow)

    # --- 3. TRAINABLE ADAPTATION CONTAINER (Layer 4 + Output, y: 11.5 - 36.5) ---
    rect_train = patches.FancyBboxPatch(
        (4, 11.5), 92, 25.0,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        linewidth=1.1, edgecolor=c_peach_border, facecolor=c_peach_bg
    )
    ax.add_patch(rect_train)

    # Container Header
    ax.text(8, 34.2, "Target Climate Adaptation Head (Trainable)", 
            fontsize=9.2, fontweight='bold', color='#9C4221')
    ax.text(92, 34.2, r"$\theta_{\mathrm{train}} =$ 2,081 params", 
            fontsize=8.5, fontweight='bold', ha='right', color='#9C4221')
    ax.text(8, 32.0, r"Fine-tuned on Cornell (Zone 6A) target records ($\eta = 10^{-4}$); adapts to sub-zero heating",
            fontsize=7.4, style='italic', color=c_text_muted)

    # Layer 4 Card (y: 22.8 - 30.3)
    r_l4 = patches.FancyBboxPatch(
        (7, 22.8), 86, 7.5,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.9, edgecolor=c_peach_border, facecolor=c_box_bg
    )
    ax.add_patch(r_l4)
    ax.text(10, 27.8, "Dense Layer 4: 32 Units", fontsize=8.8, fontweight='bold', color=c_text_dark)
    ax.text(10, 25.5, r"$\mathbf{h}_4 = \mathrm{ReLU}(\mathbf{W}_4 \mathbf{h}_3 + \mathbf{b}_4)$ $\cdot$ Calibrates local heating-degree-day response", 
            fontsize=7.5, color=c_text_muted)
    ax.text(10, 23.6, r"Output dimension: $\mathbf{h}_4 \in \mathbb{R}^{32}$", fontsize=7.2, color='#4A5568')
    ax.text(90, 27.8, "2,080 weights", ha='right', fontsize=8.0, fontweight='bold', color='#C05621')
    ax.text(90, 25.0, r"$\mathbf{W}_4 \in \mathbb{R}^{32 \times 64},\ \mathbf{b}_4 \in \mathbb{R}^{32}$", ha='right', fontsize=7.2, color=c_text_muted)

    # Arrow L4 -> Linear Readout
    ax.annotate("", xy=(50, 19.5), xytext=(50, 22.8),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=4.0", lw=0.9, color=c_arrow))

    # Readout Layer Card (y: 13.0 - 19.5)
    r_out = patches.FancyBboxPatch(
        (7, 13.0), 86, 6.5,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.9, edgecolor=c_peach_border, facecolor=c_box_bg
    )
    ax.add_patch(r_out)
    ax.text(10, 17.2, "Linear Energy Readout: 1 Unit", fontsize=8.8, fontweight='bold', color=c_text_dark)
    ax.text(10, 14.8, r"$\tilde{y} = \mathbf{w}_{\mathrm{out}}^\top \mathbf{h}_4 + b_{\mathrm{out}}$ $\cdot$ Produces log-transformed energy estimate", 
            fontsize=7.5, color=c_text_muted)
    ax.text(90, 17.2, "33 weights", ha='right', fontsize=8.0, fontweight='bold', color='#C05621')
    ax.text(90, 14.8, r"$\mathbf{w}_{\mathrm{out}} \in \mathbb{R}^{1 \times 32},\ b_{\mathrm{out}} \in \mathbb{R}$", ha='right', fontsize=7.2, color=c_text_muted)

    # Arrow Readout -> Inversion Block
    ax.annotate("", xy=(50, 7.8), xytext=(50, 11.5),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=4.5", lw=1.0, color=c_arrow))
    ax.text(53, 9.6, r"$\tilde{y} = \log(1+y)$", fontsize=7.6, va='center', color=c_arrow)

    # --- 4. PREDICTION / INVERSION BLOCK (y: 1.0 - 7.8) ---
    rect_pred = patches.FancyBboxPatch(
        (6, 1.0), 88, 6.8,
        boxstyle="round,pad=0.35,rounding_size=1.0",
        linewidth=1.0, edgecolor=c_mint_border, facecolor=c_mint_bg
    )
    ax.add_patch(rect_pred)
    ax.text(37, 5.2, r"Predicted Electricity Consumption: $\hat{y} = \exp(\tilde{y}) - 1\ [\mathrm{kWh}]$", 
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#1C4A27')
    ax.text(37, 2.6, r"Physical scale inversion $\cdot$ Evaluated on RMSE, CVRMSE, and $R^2$ against ground-truth metered load",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    # Mini sparkline inside Prediction block (hourly energy load profile)
    sub_ax_out = fig.add_axes([0.76, 0.018, 0.14, 0.032])
    hours = np.linspace(0, 24, 48)
    profile = 15 + 25 * np.exp(-((hours - 14)/4)**2) + 5 * np.sin(hours/2)
    sub_ax_out.plot(hours, profile, color='#276749', lw=1.2)
    sub_ax_out.fill_between(hours, profile, color='#C6F6D5', alpha=0.5)
    sub_ax_out.axis('off')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def draw_system_pipeline(output_path):
    """
    Renders wide full-page System Pipeline figure (Fig. 1 in paper).
    Split into (a) Source Domain Pre-Training and (b) Cross-Climate Adaptation,
    separated by a vertical divider line matching technical_report.pdf.
    """
    fig, ax = plt.subplots(figsize=(13.2, 5.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Palette
    c_blue_bg = '#F2F6FA'
    c_blue_border = '#688FAE'
    c_peach_bg = '#FDF6F0'
    c_peach_border = '#D6896C'
    c_mint_bg = '#EFF7F1'
    c_mint_border = '#559E68'
    c_gray_bg = '#F8F9FA'
    c_gray_border = '#A8B3BC'
    c_box_bg = '#FFFFFF'
    c_text_dark = '#1A2530'
    c_text_muted = '#4A5568'
    c_arrow = '#2C3E50'

    # Divider Line between (a) and (b)
    ax.plot([49.5, 49.5], [4, 96], color='#CBD5E1', lw=1.0, ls='-')

    # =========================================================================
    # (a) SOURCE DOMAIN PRE-TRAINING (x: 2 to 47.5)
    # =========================================================================
    ax.text(2.5, 96.0, "(a) Source Pre-Training & Physics Ingestion (Zone 2B Tempe, Hot-Arid)", 
            fontsize=10.5, fontweight='bold', color=c_text_dark)

    # Box 1: Multi-Modal Data Ingestion (x: 3 to 46.5, y: 73.5 to 89)
    b1 = patches.FancyBboxPatch(
        (3, 73.5), 43.5, 15.5,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.9, edgecolor=c_gray_border, facecolor=c_gray_bg
    )
    ax.add_patch(b1)
    ax.text(5, 85.5, "Multi-Modal Sensor & Operational Streams", fontsize=9.0, fontweight='bold', color=c_text_dark)
    ax.text(5, 81.8, r"$\bullet$ Meteorological: Air Temp ($T_{\mathrm{air}}$), Dew Temp ($T_{\mathrm{dew}}$), Wind Speed ($v$), Pressure ($P$)", fontsize=7.6, color=c_text_muted)
    ax.text(5, 78.5, r"$\bullet$ Temporal Dynamics: Cyclic $\sin / \cos(2\pi \cdot \mathrm{hour}/24)$ and $(2\pi \cdot \mathrm{month}/12)$", fontsize=7.6, color=c_text_muted)
    ax.text(5, 75.2, r"$\bullet$ Building Context: 8-class one-hot space $\cdot$ Target: Metered Electricity ($y$, kWh)", fontsize=7.6, color=c_text_muted)

    # Sparkline in Box 1
    sub_a1 = fig.add_axes([0.38, 0.76, 0.08, 0.08])
    t_steps = np.linspace(0, 48, 100)
    temp_curve = 25 + 12 * np.sin(2 * np.pi * t_steps / 24 - 1.5)
    sub_a1.plot(t_steps, temp_curve, color='#D97706', lw=1.0)
    sub_a1.set_title(r"$T_{\mathrm{air}}(t)$", fontsize=6.8, pad=1)
    sub_a1.axis('off')

    # Arrow Box 1 -> Box 2
    ax.annotate("", xy=(24.75, 66.5), xytext=(24.75, 73.5),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=4.5", lw=0.9, color=c_arrow))
    ax.text(26.0, 70.0, r"Raw measurements $\in \mathbb{R}^{16}$", fontsize=7.4, va='center', color=c_arrow)

    # Box 2: Preprocessing & Harmonization (x: 3 to 46.5, y: 53.0 to 66.5)
    b2 = patches.FancyBboxPatch(
        (3, 53.0), 43.5, 13.5,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.9, edgecolor=c_blue_border, facecolor=c_blue_bg
    )
    ax.add_patch(b2)
    ax.text(5, 63.5, "Physics-Aware Feature Harmonization", fontsize=9.0, fontweight='bold', color='#1B365D')
    ax.text(5, 59.8, r"$\mathbf{x} = (\mathbf{u} - \boldsymbol{\mu}_{\mathrm{src}}) / \boldsymbol{\sigma}_{\mathrm{src}}$ $\cdot$ Z-score standardization ($\mathbf{x} \in \mathbb{R}^{16}$)", fontsize=7.6, color=c_text_muted)
    ax.text(5, 56.5, r"$\tilde{y} = \log(1 + y)$ $\cdot$ Variance-stabilizing target transformation (skewness removal)", fontsize=7.6, color=c_text_muted)
    ax.text(5, 54.0, "Continuous 8,760 hourly time-series (100% full annual cycle of source climate)", fontsize=7.3, style='italic', color='#2B6CB0')

    # Arrow Box 2 -> Box 3
    ax.annotate("", xy=(24.75, 46.0), xytext=(24.75, 53.0),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=4.5", lw=0.9, color=c_arrow))
    ax.text(26.0, 49.5, r"Standardized tensors $(\mathbf{X}_{\mathrm{src}}, \mathbf{y}_{\mathrm{src}})$", fontsize=7.4, va='center', color=c_arrow)

    # Box 3: Source Deep Learning Optimization (x: 3 to 46.5, y: 25.0 to 46.0)
    b3 = patches.FancyBboxPatch(
        (3, 25.0), 43.5, 21.0,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=1.0, edgecolor=c_blue_border, facecolor=c_box_bg
    )
    ax.add_patch(b3)
    ax.text(5, 43.0, r"Deep Neural Network Optimization ($\theta_{\mathrm{source}} = 45,665\ \mathrm{params}$)", fontsize=9.0, fontweight='bold', color=c_text_dark)
    ax.text(5, 39.5, r"Dense [256] $\to$ Dense [128] $\to$ Dense [64] $\to$ Dense [32] $\to$ Linear [1]", fontsize=7.8, color=c_text_dark)
    ax.text(5, 36.2, r"$\bullet$ Loss: $\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N}\sum (\tilde{y}_i - \hat{y}_i)^2$ $\cdot$ AdamW ($\eta = 10^{-3}$, weight decay $10^{-4}$)", fontsize=7.5, color=c_text_muted)
    ax.text(5, 33.0, r"$\bullet$ Convergence: Early stopping with patience 15 epochs (MSE on validation split)", fontsize=7.5, color=c_text_muted)
    ax.text(5, 29.5, r"$\bullet$ Converged Benchmark: $R^2 = 0.9998,\ \mathrm{RMSE} = 5.92\ \mathrm{kWh},\ \mathrm{CVRMSE} = 3.84\%$", fontsize=7.6, fontweight='bold', color='#1D4ED8')
    ax.text(5, 26.5, "Captures envelope thermal mass, solar gain delays, and diurnal scheduling laws", fontsize=7.2, style='italic', color='#4A5568')

    # Arrow Box 3 -> Box 4
    ax.annotate("", xy=(24.75, 18.0), xytext=(24.75, 25.0),
                arrowprops=dict(arrowstyle="->,head_width=3,head_length=4.5", lw=0.9, color=c_arrow))

    # Box 4: Extracted Invariants (x: 3 to 46.5, y: 4.5 to 18.0)
    b4 = patches.FancyBboxPatch(
        (3, 4.5), 43.5, 13.5,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=1.0, edgecolor=c_blue_border, facecolor=c_blue_bg, linestyle='--'
    )
    ax.add_patch(b4)
    ax.text(5, 14.8, "Extracted Generalized Physics Invariants", fontsize=9.0, fontweight='bold', color='#1B365D')
    ax.text(5, 11.2, r"$\theta_{\mathrm{frozen}} = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \mathbf{W}_3, \mathbf{b}_3\} = \mathbf{43,584\ parameters\ (95.4\%)}$", fontsize=7.6, fontweight='bold', color='#1E40AF')
    ax.text(5, 8.2, "Transferred across climates to provide strong inductive bias under sparse target data", fontsize=7.4, color=c_text_muted)
    ax.text(5, 5.8, r"$\to$ Transported to Target Domain (Cornell Zone 6A)", fontsize=7.4, fontweight='bold', color='#9C4221')

    # Connecting Arrow from (a) Box 4 to (b) Transfer Path
    ax.annotate("", xy=(52.5, 55.0), xytext=(46.5, 11.0),
                arrowprops=dict(arrowstyle="->,head_width=3.5,head_length=5", lw=1.2, color='#2563EB',
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(50.5, 28.0, r"$\theta_{\mathrm{frozen}}$ Transfer", fontsize=7.6, fontweight='bold', color='#2563EB', rotation=65)

    # =========================================================================
    # (b) TARGET ADAPTATION & BENCHMARKING (x: 52 to 97.5)
    # =========================================================================
    ax.text(52.5, 96.0, "(b) Cross-Climate Adaptation & Evaluation (Zone 6A Cornell, Cold-Humid)", 
            fontsize=10.5, fontweight='bold', color=c_text_dark)

    # Box Target Ingestion & Data Scarcity Subsampling (x: 52.5 to 97, y: 76.5 to 90)
    bt_in = patches.FancyBboxPatch(
        (52.5, 76.5), 44.5, 13.5,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.9, edgecolor=c_gray_border, facecolor=c_gray_bg
    )
    ax.add_patch(bt_in)
    ax.text(54.5, 86.8, "Target Domain Ingestion & Data-Scarcity Sweep", fontsize=9.0, fontweight='bold', color=c_text_dark)
    ax.text(54.5, 83.2, r"Target Setting: Cornell University (Zone 6A Cold-Humid) $\cdot$ Severe sub-zero heating regime", fontsize=7.6, color=c_text_muted)
    ax.text(54.5, 79.8, r"Data-Scarcity Sweep: $k \in \{5\%, 10\%, 20\%, 30\%, 50\%, 100\%\}$ target availability", fontsize=7.6, fontweight='bold', color='#9C4221')
    ax.text(54.5, 77.2, r"$5\% \equiv 438\ \mathrm{hours}\ (\approx 18\ \mathrm{days})$ $\longleftrightarrow$ $100\% \equiv 8,760\ \mathrm{hours}\ (1\ \mathrm{full\ year})$", fontsize=7.3, style='italic', color='#4A5568')

    # Diamond Split Decision Node (x: 74.75, y: 69.5)
    d_poly = patches.Polygon([[74.75, 73.0], [78.5, 69.5], [74.75, 66.0], [71.0, 69.5]], 
                             facecolor='#FEF3C7', edgecolor='#D97706', lw=0.9)
    ax.add_patch(d_poly)
    ax.text(74.75, 69.5, "Strategy", ha='center', va='center', fontsize=7.0, fontweight='bold', color='#92400E')

    # Arrow from Target Ingestion down to Diamond
    ax.annotate("", xy=(74.75, 73.0), xytext=(74.75, 76.5),
                arrowprops=dict(arrowstyle="->,head_width=2.5,head_length=3.5", lw=0.8, color=c_arrow))

    # Split Arrows from Diamond to Path 1 (Transfer) and Path 2 (Scratch)
    ax.annotate("", xy=(68.0, 61.5), xytext=(71.0, 69.5),
                arrowprops=dict(arrowstyle="->,head_width=2.5,head_length=4", lw=0.9, color='#1E40AF'))
    ax.text(67.5, 66.0, "Transfer (TL)", fontsize=7.2, fontweight='bold', color='#1E40AF', ha='right')

    ax.annotate("", xy=(82.0, 61.5), xytext=(78.5, 69.5),
                arrowprops=dict(arrowstyle="->,head_width=2.5,head_length=4", lw=0.9, color='#C05621'))
    ax.text(82.5, 66.0, "Scratch (SCR)", fontsize=7.2, fontweight='bold', color='#C05621', ha='left')

    # Path 1: Transfer Learning Box (x: 52.5 to 73.5, y: 41.5 to 61.5)
    b_tl = patches.FancyBboxPatch(
        (52.5, 41.5), 21.0, 20.0,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=1.0, edgecolor=c_blue_border, facecolor=c_blue_bg
    )
    ax.add_patch(b_tl)
    ax.text(54.0, 58.8, "Transfer Learning (TL)", fontsize=8.6, fontweight='bold', color='#1E40AF')
    ax.text(54.0, 55.6, r"$\bullet$ Freeze: $\theta_{\mathrm{frozen}}$ (Layers 1--3)", fontsize=7.3, color=c_text_dark)
    ax.text(54.0, 53.0, r"$\bullet$ Train: $\theta_{\mathrm{train}} = \mathbf{2,081\ params}$", fontsize=7.3, fontweight='bold', color='#1E40AF')
    ax.text(54.0, 50.4, "  (Dense 4 [32] + Readout [1])", fontsize=7.1, color=c_text_muted)
    ax.text(54.0, 47.6, r"$\bullet$ Learning rate: $\eta = 10^{-4}$ (AdamW)", fontsize=7.2, color=c_text_muted)
    ax.text(54.0, 45.0, r"$\bullet$ Convergence: 8--12 epochs", fontsize=7.3, color='#15803D')
    ax.text(54.0, 42.6, r"5% Data: $R^2 = 0.9936,\ 14.2\ \mathrm{kWh}$", fontsize=7.4, fontweight='bold', color='#1E40AF')

    # Path 2: Scratch Baseline Box (x: 75.5 to 97.0, y: 41.5 to 61.5)
    b_scr = patches.FancyBboxPatch(
        (75.5, 41.5), 21.5, 20.0,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=1.0, edgecolor=c_peach_border, facecolor=c_peach_bg
    )
    ax.add_patch(b_scr)
    ax.text(77.0, 58.8, "Scratch Baseline (SCR)", fontsize=8.6, fontweight='bold', color='#C05621')
    ax.text(77.0, 55.6, r"$\bullet$ Init: Random Gaussian weights", fontsize=7.3, color=c_text_dark)
    ax.text(77.0, 53.0, r"$\bullet$ Train: $\theta = \mathbf{45,665\ params}$", fontsize=7.3, fontweight='bold', color='#C05621')
    ax.text(77.0, 50.4, "  (All 4 layers initialized from zero)", fontsize=7.1, color=c_text_muted)
    ax.text(77.0, 47.6, r"$\bullet$ Learning rate: $\eta = 10^{-3}$ (AdamW)", fontsize=7.2, color=c_text_muted)
    ax.text(77.0, 45.0, r"$\bullet$ Convergence: 35--50 epochs", fontsize=7.3, color='#B91C1C')
    ax.text(77.0, 42.6, r"5% Data: $R^2 = 0.8124,\ 17.2\ \mathrm{kWh}$", fontsize=7.4, fontweight='bold', color='#B91C1C')

    # Arrows from TL and SCR down into Evaluation & Decision Block
    ax.annotate("", xy=(63.0, 34.0), xytext=(63.0, 41.5),
                arrowprops=dict(arrowstyle="->,head_width=2.5,head_length=4", lw=0.9, color='#1E40AF'))
    ax.annotate("", xy=(86.25, 34.0), xytext=(86.25, 41.5),
                arrowprops=dict(arrowstyle="->,head_width=2.5,head_length=4", lw=0.9, color='#C05621'))

    # Box Comparative Benchmarking & Empirical Decision Gate (x: 52.5 to 97, y: 4.5 to 34.0)
    b_eval = patches.FancyBboxPatch(
        (52.5, 4.5), 44.5, 29.5,
        boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=1.0, edgecolor=c_mint_border, facecolor=c_mint_bg
    )
    ax.add_patch(b_eval)
    ax.text(54.5, 31.5, "Empirical Cross-Climate Evaluation & Decision Gate", fontsize=9.0, fontweight='bold', color='#14532D')
    ax.text(54.5, 28.2, r"$\bullet$ Extreme Scarcity (5%--20%): TL gains $+0.1812\ \Delta R^2$ (RMSE $14.18 \to 8.28\ \mathrm{kWh}$)", fontsize=7.5, color='#166534')
    ax.text(54.5, 25.2, r"$\bullet$ Empirical Crossover (30%): Scratch matches TL ($R^2 \approx 0.9981$ vs $0.9984$)", fontsize=7.5, color=c_text_muted)
    ax.text(54.5, 22.2, r"$\bullet$ Data Saturation (50%--100%): Scratch slightly outperforms by $0.0009\ R^2$", fontsize=7.5, color=c_text_muted)
    ax.text(54.5, 19.2, r"$\bullet$ Deployment Decision Rule:", fontsize=7.6, fontweight='bold', color='#1E3A8A')
    ax.text(56.5, 16.5, r"-- If local history $< 30\%$ ($\leq 2.5$ months): Deploy Pre-Trained Transfer Model", fontsize=7.3, color='#1E40AF')
    ax.text(56.5, 14.0, r"-- If local history $\geq 30\%$: Transition to Unconstrained Scratch Model", fontsize=7.3, color='#9A3412')
    ax.text(54.5, 10.5, r"Edge Efficiency: 95.4% fewer active parameters $\rightarrow 70\%$ lower RAM, $<4.2\ \mathrm{ms}$ inference", fontsize=7.4, style='italic', color='#15803D')
    ax.text(54.5, 7.5, "Enables autonomous calibration directly on embedded IoT building management gateways", fontsize=7.2, color=c_text_muted)

    # Inset Mini-Plot inside Evaluation block: Transfer vs Scratch Crossover curve!
    sub_eval = fig.add_axes([0.83, 0.075, 0.13, 0.12])
    data_pct = np.array([5, 10, 20, 30, 50, 100])
    r2_scratch = np.array([0.8124, 0.9798, 0.9958, 0.9981, 0.9992, 0.9998])
    r2_transfer = np.array([0.9936, 0.9955, 0.9978, 0.9984, 0.9987, 0.9989])
    sub_eval.plot(data_pct, r2_transfer, marker='o', ms=3, color='#1E40AF', lw=1.2, label='Transfer')
    sub_eval.plot(data_pct, r2_scratch, marker='s', ms=3, color='#C05621', lw=1.2, ls='--', label='Scratch')
    sub_eval.axvline(30, color='#6B7280', ls=':', lw=0.9)
    sub_eval.text(32, 0.84, 'Crossover\n(30%)', fontsize=6.0, color='#374151')
    sub_eval.set_xticks([5, 30, 100])
    sub_eval.set_xticklabels(['5%', '30%', '100%'], fontsize=6.2)
    sub_eval.set_yticks([0.8, 0.9, 1.0])
    sub_eval.set_yticklabels(['0.8', '0.9', '1.0'], fontsize=6.2)
    sub_eval.set_ylabel(r"$R^2$", fontsize=6.8, labelpad=1)
    sub_eval.legend(fontsize=6.0, loc='lower right', framealpha=0.8, handlelength=1.2)
    sub_eval.tick_params(axis='both', which='both', labelsize=6.0, pad=1)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    os.makedirs('research paper/figures', exist_ok=True)
    draw_ann_architecture('research paper/figures/ANN_Architecture.png')
    draw_system_pipeline('research paper/figures/System_Pipeline.png')
