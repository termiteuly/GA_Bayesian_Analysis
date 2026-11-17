#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 synthetic reporting: GA-only versus GA+Bayesian (tau=.5,.6,.7,.8).

This module NEVER uses FIM as a reporting gate. It loads unchanged V5
synthetic results (directories or ZIP files) and evaluates *GA point-recovery*
under the same truth-defined factor-of-two criterion for every strategy.

Bayesian gate = NUTS diagnostics PASS AND, for EACH selected parameter,
P(|ln(theta_i)-ln(GA_i)| <= ln(2) | data) >= tau.
A full-vector dataset passes only if ALL FOUR parameter gates pass.

Ground truth is used only to score synthetic recovery, NEVER to decide gates.
S1-S4 are oracle-truth FIM strata from synthetic design, NOT real-data labels.
FIM diagnostics are handled by dep_fim_diagnostics_v5.py, independently.
"""
from __future__ import annotations
import argparse
import io
import json
import zipfile
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARAMS = ('eps_cyto','sigma_cyto','eps_shell','sigma_shell')
THRESHOLDS = (0.5,0.6,0.7,0.8)
METHODS = ('GA-only',) + tuple(f'GA+Bayes_{t:.1f}' for t in THRESHOLDS)


def load_table(source, filename, mandatory=False):
    source=Path(source)
    if source.is_dir():
        p=source/filename
        if not p.exists():
            if mandatory: raise FileNotFoundError(p)
            return pd.DataFrame()
        return pd.read_csv(p,low_memory=False)
    with zipfile.ZipFile(source) as z:
        files=[n for n in z.namelist() if n==filename or n.endswith('/'+filename)]
        if not files:
            if mandatory:raise FileNotFoundError(f'{filename} in {source}')
            return pd.DataFrame()
        if len(files)>1:raise ValueError(f'ambiguous {filename} inside {source}: {files}')
        return pd.read_csv(io.BytesIO(z.read(files[0])),low_memory=False)


def truth_bool(values):
    return values.astype(str).str.strip().str.lower().isin(('true','1','yes','t'))


def stratum_short(value):
    for t in ('S1','S2','S3','S4'):
        if str(value).startswith(t):return t
    return str(value)


def read_source(source,seed_id):
    """Read V5 source without requiring FIM evaluation files."""
    g=load_table(source,'results_GA_datasets.csv',True)
    p=load_table(source,'results_GA_parameters.csv',True)
    n=load_table(source,'results_datasets.csv')
    np_=load_table(source,'results_parameters.csv')
    split=load_table(source,'audit/00_truth_discovery_validation_split.csv')
    if 'ga_status' in g:g=g[g.ga_status.astype(str).eq('ok')].copy()
    if g.empty:raise ValueError(f'{source}: no completed GA datasets')
    for obj in (g,p,n,np_,split):
        if obj.empty:continue
        obj['seed_id']=str(seed_id)
        if 'dataset_id' in obj:obj['dataset_key']=str(seed_id)+'::'+obj.dataset_id.astype(str)
        if 'truth_id' in obj:obj['truth_key']=str(seed_id)+'::'+obj.truth_id.astype(str)
    needed=['dataset_key','truth_key','dataset_id','truth_id','stratum','noise_multiplier']
    for c in needed:
        if c not in g:raise ValueError(f'{source}: missing GA dataset column {c}')
    D=g[needed].copy()
    D['seed_id']=str(seed_id)
    D['stratum']=D.stratum.map(stratum_short)
    if D.dataset_key.duplicated().any():raise ValueError('Duplicate GA dataset rows')
    if not split.empty:
        if 'split' not in split:raise ValueError('truth split missing split column')
        t=split[['truth_key','split']].drop_duplicates('truth_key')
        D=D.merge(t,on='truth_key',how='left',validate='many_to_one')
        if D.split.isna().any():raise ValueError('Stored truth-level split incomplete')
    else:
        D['split']='NOT_AVAILABLE'
    P=p[p.parameter.isin(PARAMS)].copy()
    req=['dataset_key','parameter','true_value','ga_estimate']
    for col in req:
        if col not in P:raise ValueError(f'{source}: missing GA parameter column {col}')
    P=P[P.dataset_key.isin(D.dataset_key)].copy()
    if P.duplicated(['dataset_key','parameter']).any():raise ValueError('Duplicate GA parameter records')
    if (P.groupby('dataset_key').parameter.nunique()!=4).any() or len(P)!=4*len(D):
        raise ValueError(f'{source}: each GA dataset must contain exactly four parameter rows')
    tv=pd.to_numeric(P.true_value,errors='coerce').to_numpy(float)
    gv=pd.to_numeric(P.ga_estimate,errors='coerce').to_numpy(float)
    if np.any(~np.isfinite(tv)|~np.isfinite(gv)|(tv<=0)|(gv<=0)):
        raise ValueError('GA/truth values must be positive and finite')
    P['ga_within_factor2']=np.abs(np.log(gv/tv)) <= (np.log(2)+1e-12)
    D['ga_all_4_factor2']=D.dataset_key.map(P.groupby('dataset_key').ga_within_factor2.all()).fillna(False).astype(bool)
    D['mcmc_pass']=False
    D['mcmc_status']='NOT_RUN'
    if not n.empty:
        if n.dataset_key.duplicated().any():raise ValueError('Duplicate NUTS dataset rows')
        if 'mcmc_pass' in n:
            n=n[['dataset_key','mcmc_pass'] + (['status'] if 'status' in n else [])].copy()
            n['mcmc_pass']=truth_bool(n.mcmc_pass.fillna(False))
            if 'status' in n:n['mcmc_pass'] &= n.status.astype(str).eq('ok')
            D=D.drop(columns=['mcmc_pass','mcmc_status']).merge(n,on='dataset_key',how='left',validate='one_to_one')
            D['mcmc_status']=np.where(D.mcmc_pass.isna(),'NOT_RUN',np.where(D.mcmc_pass,'PASS','FAIL'))
            D['mcmc_pass']=D.mcmc_pass.fillna(False).astype(bool)
    if not np_.empty:
        good=np_[np_.parameter.isin(PARAMS)].copy()
        if good.duplicated(['dataset_key','parameter']).any():raise ValueError('Duplicate NUTS parameter records')
        fields=['dataset_key','parameter','posterior_prob_recovery_GA_center','log10_post_q025',
                'log10_post_q975','post_median','posterior_ci_95_coverage_truth','posterior_ci_width_percent_prior']
        fields=[x for x in fields if x in good]
        fields += [x for x in good if x.startswith('v5_') and x not in fields]
        P=P.merge(good[fields],on=['dataset_key','parameter'],how='left',validate='one_to_one')
    if 'posterior_prob_recovery_GA_center' not in P:P['posterior_prob_recovery_GA_center']=np.nan
    P['posterior_prob_recovery_GA_center']=pd.to_numeric(P.posterior_prob_recovery_GA_center,errors='coerce')
    valid=P.posterior_prob_recovery_GA_center.dropna()
    if ((valid<0)|(valid>1)).any():raise ValueError('Posterior probabilities outside [0,1]')
    P=P.drop(columns=[x for x in ('stratum','split','noise_multiplier','mcmc_pass','mcmc_status') if x in P.columns])
    P=P.merge(D[['dataset_key','stratum','split','noise_multiplier','mcmc_pass','mcmc_status']],on='dataset_key',how='left',validate='many_to_one')
    for c in ('truth_key','seed_id','dataset_id','truth_id'):
        if c not in P:P[c]=P.dataset_key.map(D.set_index('dataset_key')[c])
    bad=P[P.mcmc_pass & P.posterior_prob_recovery_GA_center.isna()]
    if len(bad):
        warnings.warn(f'{len(bad)} NUTS-pass parameter records missing GA-centered posterior mass; Bayesian gates will reject these records.')
    return D,P


def make_reporting(D,P):
    D=D.copy().set_index('dataset_key')
    P=P.copy()
    P['GA-only']=True
    for tau in THRESHOLDS:
        P[f'GA+Bayes_{tau:.1f}']=P.mcmc_pass & P.posterior_prob_recovery_GA_center.ge(tau)
    grp=P.groupby('dataset_key',sort=False)
    for m in METHODS:
        D[m]=grp[m].all().reindex(D.index,fill_value=False).astype(bool)
        D['n_accepted_'+m]=grp[m].sum().reindex(D.index,fill_value=0).astype(int)
        D['n_correct_accepted_'+m]=(P[m]&P.ga_within_factor2).groupby(P.dataset_key).sum().reindex(D.index,fill_value=0).astype(int)
    return D.reset_index(),P


def summarize_one(y,accepted,clusters=None,bootstrap=0,seed=2026):
    y=np.asarray(y,dtype=bool);a=np.asarray(accepted,dtype=bool)
    if len(y)!=len(a):raise ValueError('Mismatched arrays')
    tp=int(np.sum(a&y));fp=int(np.sum(a&~y));fn=int(np.sum(~a&y));tn=int(np.sum(~a&~y))
    n=int(len(y));na=int(a.sum())
    result=dict(n=n,accepted=na,TP=tp,FP=fp,FN=fn,TN=tn,
        accept_rate=na/n if n else np.nan,
        accepted_accuracy=tp/na if na else np.nan,
        false_accept_fraction=fp/na if na else np.nan,
        correct_yield=tp/n if n else np.nan,
        incorrect_yield=fp/n if n else np.nan)
    if bootstrap and clusters is not None and n>0:
        k=pd.Categorical(clusters).codes
        K=k.max()+1
        a_by=np.bincount(k,weights=a,minlength=K)
        tp_by=np.bincount(k,weights=a&y,minlength=K)
        rng=np.random.default_rng(seed)
        sim=rng.integers(0,K,size=(bootstrap,K))
        ns=a_by[sim].sum(axis=1);ns_tp=tp_by[sim].sum(axis=1)
        acc=np.divide(ns_tp,ns,out=np.full(len(ns),np.nan),where=ns>0)
        v=acc[np.isfinite(acc)]
        result['accuracy_cluster_bootstrap_CI_low95']=float(np.quantile(v,.025)) if len(v) else np.nan
        result['accuracy_cluster_bootstrap_CI_high95']=float(np.quantile(v,.975)) if len(v) else np.nan
    return result


def prepare_summaries(D,P,bootstrap):
    ds=[];pr=[];params=[];noise=[];flow=[]
    for denom in ('ALL_ATTEMPTS','NUTS_PASS_ONLY'):
        dd_base=D if denom=='ALL_ATTEMPTS' else D[D.mcmc_pass]
        pp_base=P if denom=='ALL_ATTEMPTS' else P[P.mcmc_pass]
        for st in ('ALL','S1','S2','S3','S4'):
            dd=dd_base if st=='ALL' else dd_base[dd_base.stratum==st]
            pp=pp_base if st=='ALL' else pp_base[pp_base.stratum==st]
            if len(dd)==0:continue
            for m in METHODS:
                ds.append(dict(denominator=denom,stratum=st,method=m,n_truths=dd.truth_key.nunique(),
                   **summarize_one(dd.ga_all_4_factor2,dd[m],dd.truth_key,bootstrap)))
                pr.append(dict(denominator=denom,stratum=st,method=m,n_truths=pp.truth_key.nunique(),
                   **summarize_one(pp.ga_within_factor2,pp[m],pp.truth_key,bootstrap)))
                for param,z in pp.groupby('parameter'):
                    params.append(dict(denominator=denom,stratum=st,parameter=param,method=m,
                        **summarize_one(z.ga_within_factor2,z[m])))
    for (seed,nm),dd in D.groupby(['seed_id','noise_multiplier'],dropna=False):
        for m in METHODS:
            noise.append(dict(seed_id=seed,noise_multiplier=nm,method=m,
                **summarize_one(dd.ga_all_4_factor2,dd[m])))
    for st in ('ALL','S1','S2','S3','S4'):
        z=D if st=='ALL' else D[D.stratum==st]
        if z.empty:continue
        for m in METHODS:
            flow.append(dict(stratum=st,method=m,
                 **summarize_one(z.ga_all_4_factor2,z[m])))
    return (pd.DataFrame(ds),pd.DataFrame(pr),pd.DataFrame(params),
            pd.DataFrame(noise),pd.DataFrame(flow))


def coverage_exact95(P):
    need=('log10_post_q025','log10_post_q975')
    if any(x not in P for x in need):return pd.DataFrame()
    p=P[P.mcmc_pass].copy()
    lo=pd.to_numeric(p.log10_post_q025,errors='coerce')
    hi=pd.to_numeric(p.log10_post_q975,errors='coerce')
    truth=np.log10(p.true_value.astype(float))
    p=p[lo.notna()&hi.notna()&(lo<=hi)].copy()
    if p.empty:return pd.DataFrame()
    p['covered']=(np.log10(p.true_value)>=pd.to_numeric(p.log10_post_q025)) & (
       np.log10(p.true_value)<=pd.to_numeric(p.log10_post_q975))
    rows=[]
    for st in ('ALL','S1','S2','S3','S4'):
        ss=p if st=='ALL' else p[p.stratum==st]
        if ss.empty:continue
        for param,z in [('ALL',ss)]+list(ss.groupby('parameter')):
            rows.append(dict(stratum=st,parameter=param,n=len(z),n_truths=z.truth_key.nunique(),
                nominal=.95,empirical_coverage=float(z.covered.mean()),
                deviation_from_nominal=float(z.covered.mean()-.95),
                source='Original NUTS Q025/Q975; exact equal-tailed CI inclusion'))
    return pd.DataFrame(rows)



def qgrid_exact(P,D,out):
    """Exact empirical versus nominal coverage from original V5 online NUTS quantiles.

    When files lack aligned posterior draws/online q-specific summaries, DO NOT
    fabricate exact results via a Gaussian proxy. A separate 95% exact table is
    still produced from stored Q025/Q975 if available.
    """
    needed=[]
    for n in range(1,10):
        tag=f'q{n}0'
        needed.extend([f'v5_equal_tail_lo_log10_{tag}',f'v5_equal_tail_hi_log10_{tag}',
                       f'v5_ga_centered_radius_log10_{tag}'])
    if not all(c in P for c in needed):
        (out/'QGRID_COVERAGE_STATUS.txt').write_text(
            'Exact 10-90% coverage unavailable: V5 online posterior q-specific summaries not present. '
            '95% coverage is separately computed from original posterior Q025/Q975. '
            'A quantile-matched approximation is intentionally NOT silently substituted.\n',encoding='utf8')
        return pd.DataFrame()
    base=P[P.mcmc_pass].copy()
    base['truth_log10']=np.log10(base.true_value.to_numpy(float))
    base['ga_log10']=np.log10(base.ga_estimate.to_numpy(float))
    ds=D.set_index('dataset_key')
    rows=[]
    for th in THRESHOLDS:
        label=f'GA+Bayes_{th:.1f}'
        passes=base.dataset_key.map(ds[label]).fillna(False).to_numpy(bool)
        z=base[passes]
        if z.empty:continue
        for n in range(1,10):
            tag=f'q{n}0'
            lo=pd.to_numeric(z[f'v5_equal_tail_lo_log10_{tag}'],errors='coerce').to_numpy(float)
            hi=pd.to_numeric(z[f'v5_equal_tail_hi_log10_{tag}'],errors='coerce').to_numpy(float)
            radius=pd.to_numeric(z[f'v5_ga_centered_radius_log10_{tag}'],errors='coerce').to_numpy(float)
            gt=z.truth_log10.to_numpy(float);gc=z.ga_log10.to_numpy(float)
            for kind,mask in [('equal_tailed', (gt>=lo)&(gt<=hi)),
                              ('ga_centered',(np.abs(gt-gc)<=radius))]:
                valid=(np.isfinite(lo)&np.isfinite(hi)&(lo<=hi) if kind=='equal_tailed' else
                       np.isfinite(radius)&(radius>=0))
                zz=z[valid]
                if zz.empty:continue
                for st in ('ALL','S1','S2','S3','S4'):
                    take=zz if st=='ALL' else zz[zz.stratum==st]
                    if take.empty:continue
                    selected_index=take.index
                    # `mask` is relative to z, so use positional matching through z.index.
                    coverage=pd.Series(mask,index=z.index).loc[selected_index]
                    for prm,g in [('ALL',take)]+list(take.groupby('parameter')):
                        qq=coverage.loc[g.index]
                        rows.append(dict(stratum=st,parameter=prm,gate=label,nominal=n/10,
                          interval_kind=kind,n=len(g),n_truths=g.truth_key.nunique(),
                          empirical=float(qq.mean()),gap=float(qq.mean()-n/10),
                          interval_source='EXACT saved online NUTS quantile radii/bounds'))
    S=pd.DataFrame(rows)
    if S.empty:
        (out/'QGRID_COVERAGE_STATUS.txt').write_text('Online q-grid fields present, but no eligible finite values.\n',encoding='utf8')
        return S
    S.to_csv(out/'09_exact_qgrid_empirical_vs_nominal.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(11,9),constrained_layout=True,sharex=True,sharey=True)
    for ax,th in zip(axs.flat,THRESHOLDS):
        label=f'GA+Bayes_{th:.1f}'
        data=S[(S.stratum=='S1')&(S.parameter=='ALL')&(S.gate==label)]
        ax.plot([0,1],[0,1],'--',color='gray',label='Nominal')
        for kind,z in data.groupby('interval_kind'):
            ax.plot(z.nominal,z.empirical,'o-',label=kind)
        ax.set(xlim=(0,1),ylim=(0,1.04),title=f'S1: full-vector Bayesian gate {th:.1f}',
               xlabel='Nominal posterior mass',ylabel='Empirical truth coverage')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Exact NUTS empirical vs nominal coverage (selected GA+Bayesian cases)')
    fig.savefig(out/'03_exact_qgrid_nominal_vs_empirical.png',dpi=180);plt.close(fig)
    (out/'QGRID_COVERAGE_STATUS.txt').write_text(
        'Exact q-grid from stored V5 online posterior quantiles (not a Gaussian proxy). '
        'Coverage is conditional on threshold selection; nominal reference is not guaranteed after selection.\n',encoding='utf8')
    return S


def probability_calibration(P,out):
    """Bayesian posterior GA-centered factor-2 mass as forecast of GA truth recovery."""
    z=P[P.mcmc_pass & P.posterior_prob_recovery_GA_center.notna()].copy()
    if z.empty:return
    z['bayes_p']=z.posterior_prob_recovery_GA_center.astype(float)
    rows=[];bins=[]
    for st in ('ALL','S1','S2','S3','S4'):
        data=z if st=='ALL' else z[z.stratum==st]
        if data.empty:continue
        for prm,g in [('ALL',data)]+list(data.groupby('parameter')):
            pp=g.bayes_p.to_numpy(float);yy=g.ga_within_factor2.to_numpy(float)
            rows.append(dict(stratum=st,parameter=prm,n=len(g),n_truths=g.truth_key.nunique(),
                brier=float(np.mean((pp-yy)**2)),mean_predicted_p=float(pp.mean()),
                observed_GA_factor2=float(yy.mean())))
            idx=np.minimum(np.floor(pp*10).astype(int),9)
            for i in range(10):
                m=idx==i
                if m.any():bins.append(dict(stratum=st,parameter=prm,bin=i,n=int(m.sum()),
                    average_probability=float(pp[m].mean()),actual_GA_factor2=float(yy[m].mean())))
    pd.DataFrame(rows).to_csv(out/'10_Bayesian_probability_Brier.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'11_Bayesian_probability_calibration_bins.csv',index=False)


def nuts_diagnostics(D,out):
    rows=[]
    for st in ('ALL','S1','S2','S3','S4'):
        z=D if st=='ALL' else D[D.stratum==st]
        if not len(z):continue
        rows.append(dict(stratum=st,n_dataset=len(z),n_truths=z.truth_key.nunique(),
           NUTS_pass=int(z.mcmc_pass.sum()),NUTS_fail=int((z.mcmc_status=='FAIL').sum()),
           NUTS_not_run=int((z.mcmc_status=='NOT_RUN').sum()),
           NUTS_pass_rate=float(z.mcmc_pass.mean())))
    pd.DataFrame(rows).to_csv(out/'12_NUTS_diagnostics_by_stratum.csv',index=False)


def exact95_by_reporting(P,D,out):
    if not all(x in P for x in ('log10_post_q025','log10_post_q975')):return
    z=P[P.mcmc_pass].copy()
    if z.empty:return
    lo=pd.to_numeric(z.log10_post_q025,errors='coerce')
    hi=pd.to_numeric(z.log10_post_q975,errors='coerce')
    z=z[lo.notna()&hi.notna()&(lo<=hi)].copy()
    if z.empty:return
    z['truth_in_CI95']=(np.log10(z.true_value)>=z.log10_post_q025)&(
                         np.log10(z.true_value)<=z.log10_post_q975)
    source=D.set_index('dataset_key')
    rows=[]
    for m in METHODS:
        pass_mask=z.dataset_key.map(source[m]).fillna(False)
        zz=z[pass_mask]
        for st in ('ALL','S1','S2','S3','S4'):
            sub=zz if st=='ALL' else zz[zz.stratum==st]
            for prm,g in [('ALL',sub)]+list(sub.groupby('parameter')):
                if g.empty:continue
                rows.append(dict(gate=m,stratum=st,parameter=prm,n=len(g),n_truths=g.truth_key.nunique(),
                    nominal=.95,empirical=float(g.truth_in_CI95.mean()),
                    caution='Selection-conditional; no guarantee of nominal 95% after gating'))
    if rows:pd.DataFrame(rows).to_csv(out/'13_exact95_coverage_conditional_on_reporting.csv',index=False)


def plots(dataset,parameter,out):
    d=dataset[(dataset.denominator=='ALL_ATTEMPTS') & (dataset.stratum=='ALL')]
    p=parameter[(parameter.denominator=='ALL_ATTEMPTS') & (parameter.stratum=='ALL')]
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,z,title in ((axs[0],d,'Full-vector (4/4)'),(axs[1],p,'Individual parameter')):
        for r in z.itertuples():
            ax.scatter(r.correct_yield*100,r.incorrect_yield*100,s=60)
            ax.annotate(r.method,(r.correct_yield*100,r.incorrect_yield*100),
                        textcoords='offset points',xytext=(4,3),fontsize=8)
        ax.set(xlabel='Correct accepted / all attempts (%)',
               ylabel='Incorrect accepted / all attempts (%)',title=title)
        ax.grid(alpha=.2)
    fig.savefig(out/'01_GA_vs_Bayesian_risk_yield.png',dpi=190);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,z,title in ((axs[0],d,'Full-vector'),(axs[1],p,'Individual parameter')):
        sub=z[z.method.str.startswith('GA+Bayes_')].copy()
        xs=[float(x.rsplit('_',1)[-1]) for x in sub.method]
        ax.plot(xs,sub.accept_rate*100,'o-',label='Acceptance / all attempts')
        ax.plot(xs,sub.accepted_accuracy*100,'s-',label='Accuracy among accepted')
        ax.plot(xs,sub.correct_yield*100,'^-',label='Correct yield / all attempts')
        ax.set(xticks=list(THRESHOLDS),xlabel='Bayesian threshold',ylabel='Percent (%)',ylim=(0,105),title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'02_Bayesian_threshold_tradeoff.png',dpi=190);plt.close(fig)


def run_analysis(sources,out,split='validation',bootstrap=300,seed_labels=None):
    out=Path(out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    if seed_labels is None:seed_labels=[f'seed{i}' for i in range(len(sources))]
    if len(seed_labels)!=len(sources) or len(set(seed_labels))!=len(seed_labels):
        raise ValueError('Unique --seed-labels must match source paths')
    chunks=[read_source(s,label) for s,label in zip(sources,seed_labels)]
    D=pd.concat([x[0] for x in chunks],ignore_index=True)
    P=pd.concat([x[1] for x in chunks],ignore_index=True)
    if split!='all':
        if split not in D.split.unique():
            raise ValueError(f'No saved {split!r} split! Supply --split all explicitly if intended.')
        D=D[D.split==split].copy();P=P[P.dataset_key.isin(D.dataset_key)].copy()
    if D.empty:raise ValueError('No data after split selection')
    D,P=make_reporting(D,P)
    ds,pr,parm,noise,flow=prepare_summaries(D,P,bootstrap)
    tables={
        '01_dataset_full_vector_GA_Bayes.csv':ds,
        '02_parameter_GA_Bayes.csv':pr,
        '03_parameter_by_stratum_GA_Bayes.csv':parm,
        '04_by_seed_noise_GA_Bayes.csv':noise,
        '05_dataset_reporting_decisions.csv':D,
        '06_parameter_reporting_decisions.csv':P,
        '07_reporting_gate_flow.csv':flow,
        '08_exact95_posterior_coverage.csv':coverage_exact95(P),
    }
    for name,frame in tables.items():
        if not frame.empty:frame.to_csv(out/name,index=False)
    plots(ds,pr,out)
    qgrid_exact(P,D,out)
    probability_calibration(P,out)
    nuts_diagnostics(D,out)
    exact95_by_reporting(P,D,out)
    meta=dict(input_sources=[str(Path(p).resolve()) for p in sources],seed_labels=seed_labels,
        split=split,n_datasets=len(D),n_truths=D.truth_key.nunique(),n_parameters=len(P),
        n_NUTS_pass=int(D.mcmc_pass.sum()),
        methods=list(METHODS),thresholds=list(THRESHOLDS),factor_of_two=2.0,
        gate='GA-centered marginal posterior mass >= threshold; ALL 4 for full-vector; NUTS pass required',
        outcome='GA point recovery relative to known synthetic truth, not posterior median recovery',
        FIM='NOT USED for reporting. Analyze separately with dep_fim_diagnostics_v5.py',
        caution='S1-S4 defined at synthetic truth (oracle); not available for unknown experimental truth',
        validation='Repeated datasets within truth are correlated; non-independent truth clusters; threshold selection here is descriptive, not prospectively validated',
        denominators='ALL_ATTEMPTS and NUTS_PASS_ONLY are separately reported')
    (out/'ANALYSIS_METADATA.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n[GA vs BAYES]')
    print(ds[(ds.denominator=='ALL_ATTEMPTS')&(ds.stratum=='ALL')][['method','n','accepted','TP','FP','accepted_accuracy','correct_yield']].to_string(index=False))
    print(f'[SAVED] {out}')
    return ds,pr


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources',nargs='+',required=True,help='V5 result folders or source ZIP files')
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--split',choices=('validation','discovery','all'),default='validation')
    ap.add_argument('--bootstrap',type=int,default=300)
    ap.add_argument('--seed-labels',nargs='+',default=None,help='Explicit names, e.g. seed0 seed1 seed2 seed3')
    a=ap.parse_args()
    if a.bootstrap<0:ap.error('bootstrap must be >=0')
    run_analysis(a.sources,a.output_dir,a.split,a.bootstrap,a.seed_labels)

if __name__=='__main__':main()
