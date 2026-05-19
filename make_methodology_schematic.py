"""
make_methodology_schematic.py — Generate a 4-step methodology diagram
for the CPUC talk. Outputs methodology_schematic.pdf and .png.

Run: python make_methodology_schematic.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def make_figure():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # ----- Header -----
    ax.text(50, 57.5,
            'Rate Design Distributional Impact Analysis',
            ha='center', fontsize=20, fontweight='bold')
    ax.text(50, 54.5,
            'PG&E  ·  SCE  ·  SDG&E   |   ≈45,000 ResStock building profiles  |  40 rate scenarios  ×  4 customer types',
            ha='center', fontsize=11, style='italic', color='#555555')

    # ----- External data sources strip -----
    ax.text(50, 50.5, 'External data sources',
            ha='center', fontsize=10, fontweight='bold', color='#333333')
    sources = [
        ('GRC filings', 7),
        ('EIA Form 861', 19),
        ('ResStock + RASS', 33),
        ('CA-wide household survey', 50),
        ('CA DG Stats', 67),
        ('CARB BEV+PHEV', 80),
        ('CEC NBT schedule', 93),
    ]
    for label, x in sources:
        ax.text(x, 48.5, label, ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor='#EEEEEE', edgecolor='#888888', linewidth=0.5))

    # ----- 4 step boxes -----
    box_w, box_h = 20, 26
    y_box = 17
    xs = [3.5, 28.5, 53.5, 78.5]
    colors = ['#3B6FB6', '#4F9D5C', '#D97A2C', '#7B4FA8']
    titles = [
        '1. Utility inputs &\nResStock benchmarking',
        '2. Rate designer',
        '3. Technology adoption',
        '4. Bill calculator',
    ]
    bullets = [
        ('IOU GRC filings:\n'
         '  res. revenue, T&D,\n'
         '  rate base, ROE, wildfire\n\n'
         'EIA 861 customer counts\n\n'
         'ResStock parquets:\n'
         '  8760 hourly load\n'
         '  metadata (income,\n'
         '  home type, CZ, PUMA)\n\n'
         'RASS calibration of\n'
         'consumption by CZ'),
        ('Compute from sample:\n'
         '  R_sample, R_gross_vol,\n'
         '  BL_total, n_CARE,\n'
         '  n_nonCARE\n\n'
         'Apply 3 policy levers:\n'
         '  • Fixed charge % of\n'
         '    T&D allocation\n'
         '  • Wildfire removal\n'
         '  • ROE reduction (pp)\n\n'
         'Revenue-neutral by\nconstruction:\n'
         '  s = (R_vol + BL) /\n      R_gross_vol\n\n'
         '→ 40 rate scenarios/IOU'),
        ('Survey-derived rate:\n'
         '  π(adopt | I, H, Z)\n'
         '  conditional on income,\n'
         '  home type, CZ\n\n'
         'κ-calibrated to current\n'
         'CA penetration:\n'
         '  17% PV, 12% EV\n\n'
         '4 customer types:\n'
         '  • Non-adopter\n'
         '  • EV only\n'
         '  • PV + storage\n'
         '  • PV + EV + storage\n\n'
         'Battery co-located with\n'
         'every PV adopter'),
        ('Per-building bill:\n'
         '  TOU energy charges\n'
         '   − baseline credit\n'
         '   × CARE discount\n'
         '   + fixed charge\n\n'
         'PV adopters:\n'
         '  Net Billing Tariff\n'
         '  (hourly export\n'
         '  compensation)\n\n'
         'Battery dispatch via\n'
         'LP (HiGHS solver)\n\n'
         'Output: bills per\n'
         '(building × scenario ×\n'
         'customer type)'),
    ]

    for x, title, body, c in zip(xs, titles, bullets, colors):
        # box body
        box = FancyBboxPatch((x, y_box), box_w, box_h,
                             boxstyle="round,pad=0.4,rounding_size=1.2",
                             linewidth=1.8, edgecolor=c, facecolor='white')
        ax.add_patch(box)
        # title strip
        strip = FancyBboxPatch((x, y_box + box_h - 4.5), box_w, 4.5,
                               boxstyle="round,pad=0.0,rounding_size=1.2",
                               linewidth=0, edgecolor='none', facecolor=c)
        ax.add_patch(strip)
        # fill the bottom edge of strip with same color so it joins flush
        ax.add_patch(Rectangle((x, y_box + box_h - 6), box_w, 1.5,
                               facecolor=c, edgecolor='none'))
        ax.text(x + box_w/2, y_box + box_h - 2.3, title,
                ha='center', va='center', fontsize=12.5,
                fontweight='bold', color='white')
        # body bullets
        ax.text(x + 1.0, y_box + box_h - 6.5, body,
                fontsize=8.7, va='top', ha='left', color='#222222',
                family='sans-serif')

    # ----- Arrows between boxes -----
    arrow_data = [
        (xs[0] + box_w, xs[1], 'R_sample, BL_total,\nn_CARE, n_nonCARE'),
        (xs[1] + box_w, xs[2], 'TOU rates +\nfixed-charge schedule'),
        (xs[2] + box_w, xs[3], 'PV / EV / battery\nflags per building'),
    ]
    for (x1, x2, label) in arrow_data:
        y_arrow = y_box + box_h/2
        arrow = FancyArrowPatch((x1 + 0.5, y_arrow), (x2 - 0.5, y_arrow),
                                arrowstyle='-|>', mutation_scale=20,
                                lw=2.0, color='#444444')
        ax.add_patch(arrow)
        ax.text((x1 + x2)/2, y_arrow + 2.2, label, ha='center', va='bottom',
                fontsize=8.0, style='italic', color='#222222',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#BBBBBB', linewidth=0.6))

    # ----- Outputs strip -----
    out_y = 9
    out_box = FancyBboxPatch((6, out_y - 2), 88, 7,
                             boxstyle="round,pad=0.4,rounding_size=1",
                             linewidth=1.2, edgecolor='#444444',
                             facecolor='#F5F5F5')
    ax.add_patch(out_box)
    ax.text(50, out_y + 3.5, 'Distributional analysis output',
            ha='center', fontsize=11.5, fontweight='bold', color='#222222')
    ax.text(50, out_y + 1.5,
            'Bill changes, revenue neutrality, winner/loser counts — stratified by income bracket  ·  CARE status  ·  climate zone  ·  technology-adoption status',
            ha='center', fontsize=9.5, color='#333333')
    ax.text(50, out_y - 0.3,
            'Inputs to CPUC rate design proceedings (income-graduated fixed charges, wildfire cost recovery, ROE reform)',
            ha='center', fontsize=9, style='italic', color='#555555')

    # Subtle arrow from step 4 to output
    arrow = FancyArrowPatch((xs[3] + box_w/2, y_box - 0.5),
                            (xs[3] + box_w/2, out_y + 5),
                            arrowstyle='-|>', mutation_scale=18,
                            lw=1.5, color='#666666')
    ax.add_patch(arrow)

    plt.tight_layout()
    fig.savefig('methodology_schematic.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('methodology_schematic.png', bbox_inches='tight', dpi=180)
    print('Wrote methodology_schematic.pdf and .png')


if __name__ == '__main__':
    make_figure()
