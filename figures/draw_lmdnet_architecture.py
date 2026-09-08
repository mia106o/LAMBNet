"""Generate LAMB-Net architecture figure for paper."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Paper-friendly fonts (fallback if CJK missing)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# Colors
C_INPUT = '#E8F4FD'
C_ENCODER = '#D6EAF8'
C_DECODER = '#FDEBD0'
C_MSAF = '#FADBD8'
C_BEM = '#D5F5E3'
C_LOSS = '#E8DAEF'
C_ARROW = '#2C3E50'
C_BORDER = '#34495E'
C_ACCENT1 = '#2980B9'
C_ACCENT2 = '#E67E22'
C_ACCENT3 = '#27AE60'


def box(ax, xy, w, h, text, fc, ec=C_BORDER, fontsize=8, lw=1.2, alpha=1.0, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.08',
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=2)
    ax.add_patch(patch)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color='#1A1A1A', weight=weight, zorder=3,
            multialignment='center')
    return patch


def arrow(ax, start, end, style='->', color=C_ARROW, lw=1.3, connectionstyle='arc3,rad=0.0'):
    arr = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(arr)
    return arr


def section_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=11, weight='bold', color=color, ha='left', va='bottom')


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(8, 8.65, 'LAMB-Net: Long-range Adaptive Multi-scale Boundary Network',
            ha='center', va='center', fontsize=14, weight='bold', color='#1A252F')
    ax.text(8, 8.25, '长程自适应多尺度边界优化网络 · SAR 海洋溢油分割',
            ha='center', va='center', fontsize=10, color='#566573')

    # ===== Input =====
    box(ax, (0.4, 3.8), 1.5, 1.4, 'SAR\nInput\n256×256', C_INPUT, ec=C_ACCENT1, fontsize=9, bold=True)

    # ===== Encoder =====
    enc_x, enc_y, enc_w, enc_h = 2.3, 1.2, 4.2, 6.5
    enc_bg = FancyBboxPatch(
        (enc_x, enc_y), enc_w, enc_h,
        boxstyle='round,pad=0.06,rounding_size=0.15',
        linewidth=1.8, edgecolor=C_ACCENT1, facecolor=C_ENCODER, alpha=0.55, zorder=0)
    ax.add_patch(enc_bg)
    section_label(ax, enc_x + 0.15, enc_y + enc_h - 0.1, '(a) Encoder · 长程建模', C_ACCENT1)

    stage_y = [6.0, 5.0, 4.0, 3.0]
    stage_names = ['Stage 1\nC1  32×2', 'Stage 2\nC2  64×2', 'Stage 3\nC3  144×4', 'Stage 4\nC4  192×2']
    for i, (sy, sn) in enumerate(zip(stage_y, stage_names)):
        box(ax, (2.55, sy), 1.55, 0.75, sn, '#AED6F1', fontsize=7.5)
        # Natten + SS2D block
        box(ax, (4.25, sy), 1.85, 0.75,
            'Natten → SS2D\n(NATTEN Block ×N)', '#85C1E9', ec='#1F618D', fontsize=7)

    box(ax, (2.55, 2.15), 3.55, 0.65,
        'Natten: 局部邻域注意力 O(n)  |  SS2D: 四向扫描全局关联',
        '#D4E6F1', ec='#5DADE2', fontsize=7)

    arrow(ax, (1.9, 4.5), (2.55, 4.5))

    # ===== Decoder =====
    dec_x, dec_y, dec_w, dec_h = 7.0, 1.2, 5.8, 6.5
    dec_bg = FancyBboxPatch(
        (dec_x, dec_y), dec_w, dec_h,
        boxstyle='round,pad=0.06,rounding_size=0.15',
        linewidth=1.8, edgecolor=C_ACCENT2, facecolor=C_DECODER, alpha=0.55, zorder=0)
    ax.add_patch(dec_bg)
    section_label(ax, dec_x + 0.15, dec_y + dec_h - 0.1, '(b) Decoder · 多尺度 + 轮廓优化', C_ACCENT2)

    # MLP fuse
    box(ax, (7.25, 6.0), 2.0, 0.8, 'MLP Fuse\nc2+c3+c4 → F', '#F5CBA7', fontsize=8)

    # Three MMSCopE branches
    branch_x = 7.25
    branch_labels = ['P1 细尺度\n(1/4 res)', 'P2 中尺度\n(1/2 res)', 'P3 粗尺度\n(1/4 res)']
    branch_y = [4.85, 3.95, 3.05]
    for by, bl in zip(branch_y, branch_labels):
        box(ax, (branch_x, by), 1.45, 0.65, bl, '#FAD7A0', fontsize=7)

    box(ax, (7.25, 5.75), 1.45, 0.55, 'MMSCopE\n三路特征', '#EB984E', ec=C_ACCENT2, fontsize=7.5, bold=True)

    # MSAF
    box(ax, (9.0, 3.55), 1.7, 1.55,
        'MSAB\n多尺度自适应融合\n\nConcat(w·P)\ndim=256',
        C_MSAF, ec='#C0392B', fontsize=7.5, bold=True)

    # BEM
    box(ax, (9.0, 2.35), 1.7, 0.95,
        'BEM 边界增强\n1×1Conv → Sobel\n→ Sigmoid 注意力',
        C_BEM, ec=C_ACCENT3, fontsize=7.5, bold=True)

    # VSSM + output
    box(ax, (11.0, 3.55), 1.35, 1.55, 'VSSM\nBlock', '#A9DFBF', fontsize=8)
    box(ax, (11.0, 2.35), 1.35, 0.95, 'Proj +\nFeature\nFusion', '#ABEBC6', fontsize=7.5)

    # Encoder to decoder arrows (c2,c3,c4)
    for sy, ty in zip([5.35, 4.35, 3.35], [6.4, 6.4, 6.4]):
        arrow(ax, (6.1, sy), (7.25, ty), connectionstyle='arc3,rad=-0.15')
    arrow(ax, (6.1, 4.35), (7.25, 6.4), connectionstyle='arc3,rad=0.1')

    # Internal decoder flow
    arrow(ax, (8.7, 4.15), (9.0, 4.35))
    arrow(ax, (10.7, 4.35), (11.0, 4.35))
    arrow(ax, (9.85, 3.55), (9.85, 3.3))
    arrow(ax, (10.7, 2.85), (11.0, 2.85))

    # ===== Output =====
    box(ax, (12.7, 3.5), 1.5, 1.5, 'Seg\nHead\nOil Spill\nMask', '#FCF3CF', ec='#F39C12', fontsize=9, bold=True)
    arrow(ax, (12.35, 4.25), (12.7, 4.25))

    # ===== Loss =====
    loss_x, loss_y = 7.0, 0.15
    loss_bg = FancyBboxPatch(
        (loss_x, loss_y), 7.2, 0.85,
        boxstyle='round,pad=0.04,rounding_size=0.1',
        linewidth=1.5, edgecolor='#8E44AD', facecolor=C_LOSS, alpha=0.7, zorder=0)
    ax.add_patch(loss_bg)
    section_label(ax, loss_x + 0.1, loss_y + 0.85, '(c) Composite Loss · 复合损失', '#8E44AD')

    losses = [
        ('L_BCE\n区域分类', 7.3),
        ('L_Tversky\n小目标/不均衡', 9.0),
        ('L_Boundary\n轮廓像素', 10.7),
        ('L_GP\n边缘过渡顺滑', 12.4),
    ]
    for text, lx in losses:
        box(ax, (lx, 0.25), 1.45, 0.55, text, '#D7BDE2', ec='#8E44AD', fontsize=7)

    arrow(ax, (13.45, 3.5), (13.45, 1.15), lw=1.0, color='#8E44AD')
    ax.text(13.65, 2.3, 'supervise', fontsize=7, color='#8E44AD', rotation=90, va='center')

    # ===== Innovation callouts (right side) =====
    callout_x = 14.55
    callouts = [
        ('① 长程建模\nNatten+SS2D 串行', C_ACCENT1, 6.8),
        ('② 尺度自适应\nMSAF 动态权重', '#C0392B', 4.8),
        ('③ 轮廓优化\nBEM + 复合损失', C_ACCENT3, 2.8),
    ]
    for text, color, cy in callouts:
        box(ax, (callout_x, cy), 1.35, 1.0, text, 'white', ec=color, fontsize=7.5, lw=2.0, bold=True)

    # Legend strip at bottom
    legend_items = [
        mpatches.Patch(facecolor=C_ENCODER, edgecolor=C_ACCENT1, label='Encoder'),
        mpatches.Patch(facecolor=C_DECODER, edgecolor=C_ACCENT2, label='Decoder'),
        mpatches.Patch(facecolor=C_MSAF, edgecolor='#C0392B', label='MSAF'),
        mpatches.Patch(facecolor=C_BEM, edgecolor=C_ACCENT3, label='BEM'),
        mpatches.Patch(facecolor=C_LOSS, edgecolor='#8E44AD', label='Loss'),
    ]
    ax.legend(handles=legend_items, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, 'LMD-Net_architecture.png')
    pdf_path = os.path.join(out_dir, 'LMD-Net_architecture.pdf')
    svg_path = os.path.join(out_dir, 'LMD-Net_architecture.svg')

    plt.tight_layout()
    fig.savefig(png_path, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')
    print(f'Saved: {svg_path}')


if __name__ == '__main__':
    main()
