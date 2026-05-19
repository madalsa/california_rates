"""
make_methodology_schematic.py — Generate a 4-step methodology diagram
for the CPUC talk. Outputs methodology_schematic.pdf and .png.

Run: python make_methodology_schematic.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def make_figure():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # ----- Header -----
    ax.text(50, 47, 'Rate Design Distributional Impact Analysis',
            ha='center', fontsize=17, fontweight='bold')
    ax.text(50, 44.3,
            'PG&E · SCE · SDG&E   |   ≈45k ResStock buildings   |   40 rate scenarios × 4 customer types',
            ha='center', fontsize=9.5, style='italic', color='#666666')

    # ----- 4 step boxes -----
    box_w, box_h = 21, 18
    y_box = 18
    xs = [3, 28, 53, 78]
    colors = ['#3B6FB6', '#4F9D5C', '#D97A2C', '#7B4FA8']
    titles = ['1. Inputs & benchmarking',
              '2. Rate designer',
              '3. Technology adoption',
              '4. Bill calculator']
    bullets = [
        ('• IOU GRC filings\n'
         '  (revenue, T&D, ROE,\n'
         '   wildfire)\n'
         '• ResStock 8760 hourly\n'
         '  load + metadata\n'
         '• RASS calibration\n'
         '  by climate zone'),
        ('Three policy levers:\n'
         '• Fixed charge %\n'
         '   of T&D allocation\n'
         '• Wildfire removal\n'
         '• ROE reduction\n\n'
         'Revenue-neutral by\n'
         'construction → 40\n'
         'scenarios per IOU'),
        ('• Survey rate\n'
         '  π(adopt | inc, HT, CZ)\n'
         '• κ-calibrated to CA:\n'
         '   17% PV, 12% EV\n'
         '• Four customer types:\n'
         '   non-adopter, EV,\n'
         '   PV+stor, PV+EV+stor'),
        ('Per-building bill:\n'
         '• TOU energy − baseline\n'
         '  credit × CARE + FC\n'
         '• PV: Net Billing Tariff\n'
         '  hourly export prices\n'
         '• Battery dispatch:\n'
         '  LP (HiGHS)'),
    ]

    for x, title, body, c in zip(xs, titles, bullets, colors):
        box = FancyBboxPatch((x, y_box), box_w, box_h,
                             boxstyle="round,pad=0.3,rounding_size=0.8",
                             linewidth=1.5, edgecolor=c, facecolor='white')
        ax.add_patch(box)
        # title strip
        strip_h = 3.2
        ax.add_patch(Rectangle((x, y_box + box_h - strip_h), box_w, strip_h,
                               facecolor=c, edgecolor='none'))
        ax.text(x + box_w/2, y_box + box_h - strip_h/2, title,
                ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')
        # bullet body
        ax.text(x + 0.8, y_box + box_h - strip_h - 0.8, body,
                fontsize=8.3, va='top', ha='left', color='#222222')

    # ----- Arrows between boxes -----
    arrow_data = [
        (xs[0] + box_w, xs[1]),
        (xs[1] + box_w, xs[2]),
        (xs[2] + box_w, xs[3]),
    ]
    y_arrow = y_box + box_h/2
    for (x1, x2) in arrow_data:
        arrow = FancyArrowPatch((x1 + 0.3, y_arrow), (x2 - 0.3, y_arrow),
                                arrowstyle='-|>', mutation_scale=18,
                                lw=1.8, color='#555555')
        ax.add_patch(arrow)

    # ----- Output strip -----
    out_y = 8
    out_box = FancyBboxPatch((6, out_y - 2), 88, 6,
                             boxstyle="round,pad=0.3,rounding_size=0.8",
                             linewidth=1.0, edgecolor='#666666',
                             facecolor='#F5F5F5')
    ax.add_patch(out_box)
    ax.text(50, out_y + 2.2, 'Distributional output',
            ha='center', fontsize=10.5, fontweight='bold', color='#222222')
    ax.text(50, out_y + 0.2,
            'Bill changes & winner/loser counts by income  ·  CARE status  ·  climate zone  ·  technology adoption',
            ha='center', fontsize=9, color='#333333')

    # arrow from step 4 to output
    arrow = FancyArrowPatch((xs[3] + box_w/2, y_box - 0.3),
                            (xs[3] + box_w/2, out_y + 4),
                            arrowstyle='-|>', mutation_scale=15,
                            lw=1.3, color='#777777')
    ax.add_patch(arrow)

    plt.tight_layout()
    fig.savefig('methodology_schematic.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('methodology_schematic.png', bbox_inches='tight', dpi=180)
    print('Wrote methodology_schematic.pdf and .png')


if __name__ == '__main__':
    make_figure()
