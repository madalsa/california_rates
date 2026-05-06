# Assumptions & sources

Last verified: 2026-05-06. Update before each paper revision.

## Critical 2026 policy flags

These changed materially from 2024 and reshape the paper's framing:

- **Federal 25C / 25D / 30D residential clean-energy credits repealed**
  for installations / vehicles after their respective sunsets under
  One Big Beautiful Bill (P.L. 119-21, July 4, 2025). Section 30C (EVSE)
  sunsets June 30, 2026.
  Source: [IRS OBBB FAQ](https://www.irs.gov/newsroom/faqs-for-modification-of-sections-25c-25d-25e-30c-30d-45l-45w-and-179d-under-public-law-119-21-139-stat-72-july-4-2025-commonly-known-as-the-one-big-beautiful-bill-obbb)
- **CA HEAR (HEEHRA) single-family fully reserved as of 2026-02-24**;
  multifamily still open. Treat single-family base case as $0 from HEAR
  unless re-appropriated.
  Source: [TECH HEEHRA tracker](https://techcleanca.com/incentives/heehrarebates/)
- **CVRP closed Nov 2023.** No broad-market CA EV rebate exists. Only
  DCAP and Clean Cars 4 All, both income-restricted.
  Source: [CARB DCAP](https://ww2.arb.ca.gov/our-work/programs/driving-clean-assistance-program/about)
- **NBT (Net Billing Tariff, "NEM 3.0")** applies to all new PV
  interconnections after Apr 15, 2023. Average export value ~$0.05-0.08/kWh
  vs retail under NEM 2.0.
  Source: [CPUC NEM/NBT](https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/demand-side-management/customer-generation/net-energy-metering-and-net-billing)

## Capex (CA installed, 2026, pre-incentive)

| Item | Value | Range | Source |
|---|---|---|---|
| PV $/kW DC | $2,500 | $2,400-$2,600 | [EnergySage CA 2026](https://www.energysage.com/local-data/solar/ca/) |
| Battery $/kWh installed | $1,050 | $1,000-$1,100 (Powerwall 3) | [SolarReviews 2026](https://www.solarreviews.com/blog/is-the-tesla-powerwall-the-best-solar-battery-available) |
| EV premium vs ICE | $5,800 | KBB ATP gap | [Cox Auto Mar 2026 ATP](https://www.coxautoinc.com/insights/mar-2026-atp-report/) |
| L2 EVSE installed | $1,500 | $800-$2,700 | [EcoFlow 2026 guide](https://energy.ecoflow.com/us/blog/level-2-charger-installation-cost) |
| 3-ton ducted HP | $15,000 | $12-$18K (CA) | [Reliable HVAC LA 2026](https://reliablehvac.org/heat-pump-cost-in-los-angeles-and-ventura-county-2026/) |
| HPWH 50-80 gal | $5,500 | $4-$8K | [Today's Homeowner 2026](https://todayshomeowner.com/plumbing/cost/heat-pump-water-heater-cost/) |
| Induction range install | $3,500 | $2.5-$6K | [Custom Home Bay Area 2026](https://www.customhome.us/blog/induction-vs-gas-range-bay-area) |
| 200A panel upgrade | $3,500 | $2-$4.5K | [Expert Electric Group CA 2026](https://expertelectricgroup.com/electrical-panel-upgrade-california/) |

## Incentives (2026, status flagged)

### Federal (post-OBBB)

| Program | Status | Value | Source |
|---|---|---|---|
| Section 25D PV ITC | **REPEALED post-12/31/2025** | 0% (was 30%) | [IRS OBBB FAQ](https://www.irs.gov/newsroom/faqs-for-modification-of-sections-25c-25d-25e-30c-30d-45l-45w-and-179d-under-public-law-119-21-139-stat-72-july-4-2025-commonly-known-as-the-one-big-beautiful-bill-obbb) |
| Section 25D battery | **REPEALED 1/1/2026** | $0 | same |
| Section 25C HP/HPWH/panel | **EXPIRED 12/31/2025** | $0 | same |
| Section 30D EV | **REPEALED post-9/30/2025** | $0 (was $7,500) | same |
| Section 30C EVSE | Sunsets 6/30/2026 | 30% / $1,000 cap | same |
| HOMES (federal IRA, CEC-run) | **Active** (NOT repealed) | up to $8K low-income / $4K market, performance-based | [CEC IRA](https://www.energy.ca.gov/programs-and-topics/programs/inflation-reduction-act-residential-energy-rebate-programs) |
| HEAR/HEEHRA single-family | **Waitlisted 2/24/2026** | up to $14K | [TECH tracker](https://techcleanca.com/incentives/heehrarebates/) |

### California heat-pump stack (active, stackable unless noted)

| Program | HP Space | HPWH | Eligibility | Stacks? | Source |
|---|---|---|---|---|---|
| TECH Clean CA market | $1,000/sys, $2,000/home cap | $1,100-$4,300 | IOU customer | n/a | [TECH SF](https://techcleanca.com/incentives/single-family-incentives/) |
| TECH Clean CA equity | $3,500-$4,000 | $3,500-$5,700 | <80% AMI / CARE-FERA | n/a | same |
| SGIP-HPWH (CPUC) | — | $3,800 std / $4,885 LI; +$1,500 low-GWP | IOU electric+gas | **Yes** | [CPUC SGIP](https://www.cpuc.ca.gov/sgip) |
| Golden State Rebates | $50-$120 thermostat | $300 instant | any IOU residential | **Yes** | [GSR](https://goldenstaterebates.com/goldenstaterebates/rebates/heat-pump-water-heaters/) |
| BayREN Home+ | varies | $250 elec-replace / $400 fuel-sub | 9 Bay Area counties | **Yes** | [BayREN](https://www.bayren.org/news/bay-area-regional-energy-network-announces-rebate-adjustments) |
| 3C-REN (SLO/SB/Ventura) | — | $5,000 SF contractor incentive | tri-county | **Yes** | [3C-REN](https://www.3c-ren.org/frequently-asked-questions/) |
| LADWP rebate (POU) | up to $2,500/ton | varies | LADWP electric | **Yes** (POU not in TECH) | [LADWP](https://www.ladwp.com/residential-services/assistance-programs/consumer-rebate-program) |
| SMUD (POU) | $3,000 gas→elec / $1,000 elec→elec | $4,000 gas→elec / $1,000 elec→elec | SMUD electric | **Yes** | [SMUD](https://www.smud.org/Corporate/About-us/News-and-Media/2026/2026/SMUD-boosts-rebates-for-heat-pumps) |
| EBD Direct Install (CEC) | turnkey free | turnkey free | <80% AMI, priority CZs | **Replaces** TECH | [CEC EBD](https://www.energy.ca.gov/programs-and-topics/programs/equitable-building-decarbonization-program/ebd-statewide-direct) |
| ESA (CPUC IOU) | free where unsafe | free where unsafe | <=200% FPL or categorical | **Replaces** TECH | [CPUC ESA](https://www.cpuc.ca.gov/consumer-support/financial-assistance-savings-and-discounts/energy-savings-assistance) |
| LIWP (CSD) | free | limited | DAC + low-income | **Replaces** | [CSD LIWP](https://www.csd.ca.gov/Shared%20Documents/LIWP-Fact-Sheet.pdf) |

**Stacking rule used in model:** TECH + SGIP-HPWH + Golden State Rebates + REN + (POU rebate if applicable) for IOU/POU customers. For income-qualified <80% AMI in priority CZs, use EBD Direct Install (turnkey, replaces stack). For <=200% FPL with unsafe equipment, use ESA (turnkey, replaces stack).

### CA solar / battery

| Program | Status | Value | Source |
|---|---|---|---|
| SGIP storage General Market | Active | ~$200/kWh, 30 kWh cap | [SGIP metrics](https://www.selfgenca.com/home/program_metrics/) |
| SGIP storage Equity | Active | ~$850/kWh | same |
| SGIP storage Equity Resiliency | Active | ~$1,050/kWh, 80 kWh cap | same |

### CA EV rebates

| Program | Status | Value | Source |
|---|---|---|---|
| DCAP new ZEV | Active, ≤300% FPL | $7,500 + $4,500 DAC = up to $12,000 | [CARB DCAP](https://ww2.arb.ca.gov/our-work/programs/driving-clean-assistance-program/about) |
| Clean Cars 4 All — BAAQMD | Active | new $9,500 / used $7,500 / EVSE $2,000 | [CARB CC4A](https://ww2.arb.ca.gov/our-work/programs/clean-cars-4-all) |
| Clean Cars 4 All — SCAQMD | Active | new $12,000 / used $9,500 / EVSE $2,000 | same |
| Clean Cars 4 All — SJVAPCD | Active | new $9,500 / used $7,500 / EVSE $2,000 | same |
| Clean Cars 4 All — SMAQMD | Active | new $9,500 / used $7,500 / EVSE $2,000 | same |
| Clean Cars 4 All — SDAPCD | not run | n/a (DCAP only) | same |

## Fuel prices (CA, 2026)

| Item | Value | Source |
|---|---|---|
| Gasoline avg | $4.90/gal | [CEC](https://www.energy.ca.gov/estimated-gasoline-price-breakdown-and-margins), [EIA](https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_sca_w.htm) |
| PGE residential gas | $2.92/therm (Jan 2026) | [PG&E gas advisory 1/2026](https://www.pge.com/assets/pge/docs/account/rate-plans/gas-rate-advisory-0126.pdf) |
| SoCalGas (SCE territory) | $2.08/therm | [SoCalGas 1/2026](https://www.socalgas.com/sites/default/files/2026-01/SCG_GasRateAlert2026%20January.pdf) |
| SDGE gas | ~$2.10/therm | SoCalGas-tariff structure |
| Real bill escalator | 2%/yr (sensitivity 0-4%) | [Cal Advocates Q1 2025](https://www.publicadvocates.cpuc.ca.gov/-/media/cal-advocates-website/files/press-room/reports-and-analyses/242005-public-advocates-office-q1-2025-rates-report.pdf) |

## EV / vehicle

| Item | Value | Source |
|---|---|---|
| EV efficiency sedan | 4.0 mi/kWh | [Recurrent 2026](https://www.recurrentauto.com/research/most-efficient-ev) |
| EV efficiency crossover | 3.3 mi/kWh | same |
| ICE comparable MPG | 28 combined | [EPA 2024 Auto Trends](https://climateprogramportal.org/wp-content/uploads/2025/02/The-2024-EPA-Automotive-Trends-Report.pdf) |
| CA per-vehicle VMT | 10,200 mi/yr (2023) | [Caltrans](https://dot.ca.gov/programs/sustainability/sb-743/ca-vmt) |
| Household VMT default | 12,000 mi/yr | implied 2 adults / 1.8 vehicles |

## Financial

| Item | Value | Source |
|---|---|---|
| Discount rate (real, customer perspective) | 5% | [CPUC 2024 ACC v1b](https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/energy-division/documents/demand-side-management/acc-models-latest-version/updated-2024-acc-documentation-v1b.pdf) |
| Inflation | 2.5% | base case |
| Analysis horizon | 20 yr | NEM 2.0 grandfathering, PV warranty |
| PV life | 30 yr | [NREL ATB 2024](https://atb.nrel.gov/electricity/2024/residential_pv) |
| Battery life | 15 yr | [NREL ATB 2024b](https://atb.nrel.gov/electricity/2024b/residential_battery_storage) |
| Inverter replacement | year 13 | manufacturer typical |

## Export rates / EEC

We use **actual hourly utility EEC values** rather than a blended scalar.
Files (already in parent repo):

- `eec_hourly_2025.csv` — datetime, utility, eec_total ($/kWh) — 8,760 hr × 3 utilities
- `LY2025 NBT Pricing Upload MIDAS.csv` — CPUC source upload
- `PGE_Solar Billing Plan_EEC Values Price Sheet 2024/2025.pdf` — PG&E price sheets

Annual-average values computed from `eec_hourly_2025.csv` (2026-05-06):

| Utility | Avg EEC $/kWh | Source |
|---|---|---|
| PGE | $0.0968 | hourly file; PG&E EEC 2025 price sheet |
| SCE | $0.0853 | hourly file |
| SDGE | $0.0782 | hourly file |

Counterfactual / sensitivity:

| Scenario | Value | Notes |
|---|---|---|
| NEM 2.0 blended | $0.32/kWh | grandfathered customers (retail offset) |
| Flat low | $0.05/kWh | "what-if NBT was lowered" |
| Flat high | $0.15/kWh | "what-if NBT was raised" |

The full hourly profile matters for sizing because it interacts with
battery dispatch — same average can give very different NPV depending on
the within-day shape. Sizing optimizer should use the hourly file directly.

## EV acquisition scenarios

Modeled separately because the effective EV "premium" depends on
whether the household is replacing a new car vs scrapping an old gas car
under CC4A.

| Scenario | Effective premium | Eligibility |
|---|---|---|
| `new_new` (base case) | $5,800 | any household |
| `new_ev_dcap` | -$1,700 ($5,800 - $7,500) | ≤300% FPL |
| `new_ev_dcap_dac` | -$6,200 ($5,800 - $12,000) | ≤300% FPL + DAC residency |
| `scrap_replace_cc4a` | -$6,200 ($5,800 - $12,000) | income + air-district rules; old vehicle scrapped, salvage = $0 |

Sources:
- [CARB DCAP](https://ww2.arb.ca.gov/our-work/programs/driving-clean-assistance-program/about)
- [CARB Clean Cars 4 All](https://ww2.arb.ca.gov/our-work/programs/clean-cars-4-all)

CC4A rebates vary by air district (BAAQMD, SCAQMD, SJVAPCD, SMAQMD).
Use $12,000 as upper bound; actual rebate often $7,500-$12,000 depending
on income tier and replacement vehicle type.
