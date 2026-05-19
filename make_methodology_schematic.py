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
    ax.set_ylim(0, 70)
    ax.axis('off')

    # ----- Header -----
    ax.text(50, 67, 'Rate Design Distributional Impact Analysis — Methods',
            ha='center', fontsize=17, fontweight='bold')
    ax.text(50, 64.3,
            'PG&E · SCE · SDG&E   |   ≈45k ResStock buildings   |   40 rate scenarios × 4 customer types',
            ha='center', fontsize=9.5, style='italic', color='#666666')

    # ----- 4 step boxes -----
    box_w, box_h = 17, 56
    y_box = 6
    xs = [4, 28, 52, 76]
    colors = ['#3B6FB6', '#4F9D5C', '#D97A2C', '#7B4FA8']
    titles = ['1. Inputs & benchmarking',
              '2. Rate designer',
              '3. Technology adoption',
              '4. Bill calculator']

    # Each box is a list of (sub-stage label, detail string)
    stages = [
        [   # Step 1
            ('1a. IOU financials',
             'GRC filings: residential\nrevenue, T&D, ROE, rate\nbase, wildfire recovery'),
            ('1b. Customer counts',
             'EIA Form 861:\nCARE & non-CARE\nresidential accounts'),
            ('1c. ResStock load',
             '8,760-hr load profiles +\nmetadata (income, CZ,\nhome type, PUMA)'),
            ('1d. RASS calibration',
             'CZ-specific scaling of\nResStock consumption\nto match RASS survey'),
            ('1e. Baseline allowance',
             'Tier-1 kWh/day by\n(PUMA × summer/winter)\nfrom IOU tariff sheets'),
            ('1f. TOU periods',
             'Peak / mid / off-peak\nby season; wd/we split\nfor SCE summer peak'),
        ],
        [   # Step 2
            ('2a. R_sample',
             'Σ (actual-tariff bills) ×\nBUILDING_WEIGHT, on\nvalid-sample subset'),
            ('2b. R_gross_vol, BL',
             'Gross volumetric rev.\n(pre-credit) and total\nbaseline-credit pool'),
            ('2c. Policy levers',
             '• Fixed charge % of T&D\n• Wildfire removal\n• ROE reduction (pp)'),
            ('2d. Revenue target',
             'R_target = R_sample\n − wildfire_removed\n − ROE_removed'),
            ('2e. FC per customer',
             'R_fixed = R_target ·\nα_T&D · F%; allocate\nacross CARE/non-CARE'),
            ('2f. Scaled TOU rates',
             's = (R_vol + BL) /\nR_gross_vol; revenue\nneutral by construction'),
            ('→ 40 scenarios / IOU',
             '8 reported + sensitivity\nover (F, WF, ROE)\ncombinations'),
        ],
        [   # Step 3
            ('3a. Bin covariates',
             'Income (4 bands), home\ntype (SF/MF), CEC CZ\n(16 zones) per building'),
            ('3b. Survey rates π',
             'P(adopt | inc, HT, CZ)\nfrom CA-wide household\nsurvey (n = 7,182)'),
            ('3c. Fallback for sparse',
             'Cells with no respondents\n→ marginal rate over\n(income × home type)'),
            ('3d. κ-calibration',
             'Scale π so weighted\nadoption matches admin:\n17% PV, 12% EV'),
            ('3e. Probabilistic assign',
             'Seeded Bernoulli draw;\nbattery co-located with\nevery PV adopter'),
            ('→ 4 customer types',
             'Non-adopter, EV only,\nPV + storage,\nPV + EV + storage'),
        ],
        [   # Step 4
            ('4a. TOU energy charge',
             'Σ (hourly load ×\nhourly TOU rate) over\nthe 8,760-hour year'),
            ('4b. Baseline credit',
             '− min(monthly kWh,\nmonthly allowance) ×\ncredit rate ($/kWh)'),
            ('4c. CARE discount',
             '× (1 − discount) for\nCARE customers (32–35%\noff energy charges)'),
            ('4d. Fixed charge',
             '+ FC_CARE or FC_non-CARE\n(monthly × 12)'),
            ('4e. PV export (NBT)',
             'Hourly export priced\nat avoided cost (CEC\nNet Billing Tariff)'),
            ('4f. Battery dispatch',
             'LP (HiGHS) minimizes\nannual cost given load,\nPV, NBT, retail rates'),
            ('→ Bill matrix',
             '(building × scenario ×\ncustomer type) →\ndistributional analysis'),
        ],
    ]

    for x, title, sub, c in zip(xs, titles, stages, colors):
        # outer box
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

        # sub-stage cards stacked vertically
        n = len(sub)
        body_top = y_box + box_h - strip_h - 1.0
        body_bot = y_box + 1.0
        body_height = body_top - body_bot
        sub_h = body_height / n
        for i, (label, detail) in enumerate(sub):
            yi_top = body_top - i * sub_h
            yi_bot = yi_top - sub_h + 0.4
            # subtle sub-box
            sub_box = FancyBboxPatch((x + 0.5, yi_bot), box_w - 1.0, sub_h - 0.6,
                                     boxstyle="round,pad=0.12,rounding_size=0.35",
                                     linewidth=0.6, edgecolor=c, facecolor='#FAFAFA')
            ax.add_patch(sub_box)
            # sub-stage label
            ax.text(x + 0.9, yi_top - 0.6, label,
                    fontsize=7.4, fontweight='bold', va='top', ha='left', color=c)
            # detail
            ax.text(x + 0.9, yi_top - 1.9, detail,
                    fontsize=6.4, va='top', ha='left', color='#222222', linespacing=1.12)

    # ----- Arrows between boxes -----
    y_arrow = y_box + box_h/2
    for i in range(3):
        x1 = xs[i] + box_w
        x2 = xs[i+1]
        arrow = FancyArrowPatch((x1 + 0.2, y_arrow), (x2 - 0.2, y_arrow),
                                arrowstyle='-|>', mutation_scale=16,
                                lw=1.6, color='#555555')
        ax.add_patch(arrow)

    plt.tight_layout()
    fig.savefig('methodology_schematic.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('methodology_schematic.png', bbox_inches='tight', dpi=180)
    print('Wrote methodology_schematic.pdf and .png')


if __name__ == '__main__':
    make_figure()
