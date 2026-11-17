#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run GA/Bayesian reporting and independent FIM diagnostics for V5 synthetic data.

This orchestrates two INDEPENDENT analyses. FIM never enters reporting gates.
Run from existing V5 output directories/ZIPs without refitting GA or NUTS.
A STATUS JSON is always written (unless the output directory is inaccessible).
"""
from __future__ import annotations
import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path


def run_postprocess(sources, output_dir, seed_labels=None, split='all',
                    bootstrap=200, recompute_correlations=True):
    from dep_analysis_v5 import run_analysis
    from dep_fim_diagnostics_v5 import run_fim

    srcs = [Path(s).expanduser().resolve() for s in sources]
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if seed_labels is None:
        seed_labels = [f'seed{i}' for i in range(len(srcs))]
    if len(seed_labels) != len(srcs):
        raise ValueError('Number of seed labels must equal the number of source paths')

    tasks = [
        ('GA_Bayesian_reporting', out / 'v5_GA_Bayesian_reporting',
         lambda folder: run_analysis(srcs, folder, split=split, bootstrap=bootstrap,
                                     seed_labels=seed_labels)),
        ('FIM_identifiability_diagnostics', out / 'v5_FIM_diagnostics',
         lambda folder: run_fim(srcs, folder, seed_labels=seed_labels,
                                recompute=recompute_correlations)),
    ]
    status = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'sources': [str(s) for s in srcs],
        'seed_labels': seed_labels,
        'split': split,
        'reporting_methods': ['GA-only', 'GA+Bayes_0.5', 'GA+Bayes_0.6',
                              'GA+Bayes_0.7', 'GA+Bayes_0.8'],
        'FIM_as_reporting_gate': False,
        'tasks': {},
    }
    for name, folder, task in tasks:
        folder.mkdir(parents=True, exist_ok=True)
        print(f'\n[POSTPROCESS START] {name}: {folder}', flush=True)
        try:
            task(folder)
            result = {'state': 'SUCCESS', 'output_directory': str(folder)}
            print(f'[POSTPROCESS SUCCESS] {name} => {folder}', flush=True)
        except Exception as exc:
            result = {'state': 'FAILED', 'output_directory': str(folder),
                      'exception': repr(exc), 'traceback': traceback.format_exc()}
            (folder / 'ERROR.txt').write_text(result['traceback'], encoding='utf-8')
            print(f'[POSTPROCESS FAILED] {name}: {exc}\n'
                  f'  See {folder / "ERROR.txt"}', flush=True)
        status['tasks'][name] = result
        (out / 'POSTPROCESS_STATUS.json').write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', required=True,
                        help='One or more V5 result folders/ZIPs, NOT individual CSVs')
    parser.add_argument('--output-dir', required=True,
                        help='Parent of v5_GA_Bayesian_reporting and v5_FIM_diagnostics')
    parser.add_argument('--seed-labels', nargs='+', default=None)
    parser.add_argument('--split', choices=('all', 'discovery', 'validation'), default='all')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--no-recompute-correlations', action='store_true',
                        help='Use existing FIM SD data only; do not recompute JAX FIM correlations')
    args = parser.parse_args()
    status = run_postprocess(args.sources, args.output_dir, args.seed_labels,
                             args.split, args.bootstrap,
                             not args.no_recompute_correlations)
    if any(t['state'] != 'SUCCESS' for t in status['tasks'].values()):
        raise SystemExit('Postprocessing incomplete. See POSTPROCESS_STATUS.json and ERROR.txt files.')


if __name__ == '__main__':
    main()
