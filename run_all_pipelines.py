"""
run_all_pipelines.py — Run PGE, SCE, and SDGE rate analysis pipelines sequentially

Usage:
  python run_all_pipelines.py                    # full run, all 3 utilities
  python run_all_pipelines.py --test             # test mode (50 buildings each)
  python run_all_pipelines.py --utility pge      # run only PGE
  python run_all_pipelines.py --utility sce sdge # run SCE and SDGE
  python run_all_pipelines.py --stage 3          # start from stage 3
  python run_all_pipelines.py --skip-tech        # skip tech adoption stages

Run unattended (Linux):
  nohup python run_all_pipelines.py > pipeline_all.log 2>&1 &
"""

import argparse
import subprocess
import sys
import time


UTILITIES = {
    'pge': {
        'script': 'run_pge_pipeline.py',
        'label': 'PG&E',
        'baseline_dir': './Baseline_PGE',
    },
    'sce': {
        'script': 'run_sce_pipeline.py',
        'label': 'SCE',
        'baseline_dir': './Baseline_SCE',
    },
    'sdge': {
        'script': 'run_sdge_pipeline.py',
        'label': 'SDG&E',
        'baseline_dir': './Baseline_SDGE',
    },
}


def run_utility(name, info, extra_args):
    """Run a single utility pipeline as a subprocess."""
    import os
    if not os.path.exists(info['baseline_dir']):
        print(f"\n  WARNING: {info['baseline_dir']} not found — skipping {info['label']}")
        return False

    cmd = [sys.executable, info['script']] + extra_args
    print(f"\n{'='*80}")
    print(f"  RUNNING {info['label']} PIPELINE: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    status = 'OK' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f"\n  {info['label']}: {status} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Run rate analysis pipelines for all California IOUs')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (50 buildings per utility)')
    parser.add_argument('--utility', nargs='+', default=['pge', 'sce', 'sdge'],
                        choices=['pge', 'sce', 'sdge'],
                        help='Which utilities to run (default: all)')
    parser.add_argument('--stage', type=int, default=None,
                        help='Start from this stage')
    parser.add_argument('--skip-tech', action='store_true',
                        help='Skip technology adoption stages')
    parser.add_argument('--n-buildings', type=int, default=None,
                        help='Number of buildings per utility')
    args = parser.parse_args()

    # Build extra args to pass through
    extra_args = []
    if args.test:
        extra_args.append('--test')
    if args.stage is not None:
        extra_args.extend(['--stage', str(args.stage)])
    if args.skip_tech:
        extra_args.append('--skip-tech')
    if args.n_buildings is not None:
        extra_args.extend(['--n-buildings', str(args.n_buildings)])

    print("=" * 80)
    print("CALIFORNIA IOU RATE ANALYSIS — ALL UTILITIES")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Utilities: {', '.join(u.upper() for u in args.utility)}")
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"Args: {extra_args}")
    print("=" * 80)

    total_start = time.time()
    results = {}

    for name in args.utility:
        info = UTILITIES[name]
        ok = run_utility(name, info, extra_args)
        results[info['label']] = 'OK' if ok else 'FAILED'

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("ALL PIPELINES COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    for label, status in results.items():
        print(f"  {label}: {status}")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
