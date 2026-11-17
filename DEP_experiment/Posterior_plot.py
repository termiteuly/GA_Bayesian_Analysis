#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate V5 posterior KDE marginal and corner figures without rerunning NUTS.

Requires existing experiment_joint_posterior.npz (--save-posteriors) OR
experiment_idata.nc (--save-idata), and GA_FIM_parameters.csv in --results-dir.
Scientific inference results and previously generated CSV are not modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from dep_synthetic_full_v5 import PARAM_NAMES
from dep_experiment_v5_extended import _plot_corner, _plot_marginal


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=None,
                   help='Default: <results-dir>/kde_replotted, keeping original images')
    p.add_argument('--marginal-style',choices=('kde','hist+kde'),default='kde')
    p.add_argument('--marginal-kde-bandwidth',type=float,default=1.0)
    p.add_argument('--marginal-max-draws',type=int,default=8000)
    p.add_argument('--corner-max-draws',type=int,default=2500)
    p.add_argument('--seed',type=int,default=0)
    return p.parse_args()


def read_draws(results):
    npz=results/'experiment_joint_posterior.npz'
    nc=results/'experiment_idata.nc'
    if npz.is_file():
        with np.load(npz,allow_pickle=False) as data:
            return {name:np.asarray(data[name]) for name in data.files},str(npz)
    if nc.is_file():
        import arviz as az
        idata=az.from_netcdf(nc)
        samples={}
        for name in idata.posterior.data_vars:
            arr=np.asarray(idata.posterior[name])
            # (chain, draw, [dimensions...]) -> (combined draws, [dimensions...])
            samples[name]=arr.reshape((-1,)+arr.shape[2:])
        return samples,str(nc)
    raise FileNotFoundError('Neither experiment_joint_posterior.npz nor experiment_idata.nc was found. '
                            'This replot-only script requires an already SAVED posterior. '
                            'If neither was saved, rerun inference with --save-posteriors or --save-idata.')


def replot(results,output,a):
    results=Path(results)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    samples,source=read_draws(results)
    needed=['log_'+p for p in PARAM_NAMES]
    missing=[p for p in needed if p not in samples]
    if missing:
        raise KeyError('Missing dielectric posterior draws: '+', '.join(missing))
    m=np.column_stack([np.asarray(samples[k]).reshape(-1) for k in needed])
    if 'sigma_y' in samples:
        noise_model='common'
        sigma=np.asarray(samples['sigma_y']).reshape(-1)
    elif 'sigma_by_frequency' in samples:
        noise_model='per_frequency'
        sig=np.asarray(samples['sigma_by_frequency'])
        sigma=sig.reshape((-1,)+sig.shape[2:]) if sig.ndim>2 else sig
        if sigma.ndim==1 and len(sigma)%6==0:
            sigma=sigma.reshape((-1,6))
    else:
        raise KeyError('Missing sigma_y or sigma_by_frequency posterior samples')
    ga_csv=results/'GA_FIM_parameters.csv'
    if not ga_csv.is_file():
        raise FileNotFoundError(str(ga_csv))
    df=pd.read_csv(ga_csv).set_index('parameter')
    ga_theta=np.array([float(df.loc[p,'GA_estimate']) for p in PARAM_NAMES])
    diag_path=results/'NUTS_diagnostics_and_joint.json'
    if diag_path.is_file():
        diag=json.loads(diag_path.read_text(encoding='utf-8'))
    else:
        diag={'nuts_pass':False}
        print('[WARNING] NUTS diagnostics file missing; marking corner as unvalidated')
    if len(m)!=len(sigma):
        raise ValueError('Posterior noise and dielectric draws have different lengths')
    _plot_corner(output,a,m,ga_theta,diag)
    _plot_marginal(output,a,m,sigma,noise_model)
    print('[POSTERIOR SOURCE]',source)
    print('[DRAW COUNT]',len(m),'[NOISE MODEL]',noise_model)
    print('[SAVED]',output/'06_joint_posterior_corner.png')
    print('[SAVED]',output/'07_posterior_marginals.png')
    print('[NOTE] Figures only; no refitting or changes to inference/reporting decisions')


if __name__=='__main__':
    a=parse_args()
    root=a.results_dir.expanduser().resolve()
    target=a.output_dir.expanduser().resolve() if a.output_dir else root/'kde_replotted'
    replot(root,target,a)
