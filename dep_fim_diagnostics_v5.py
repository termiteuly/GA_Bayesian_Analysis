#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent V5 FIM diagnostics (NO reporting PASS/FAIL gate).

Compare FIM condition number, weakest eigenvalue, GA-location local log-SD,
and parameter-compensation correlation by synthetic stratum S1-S4.

With JAX available, recomputes 4x4 FIM at:
  (1) GA estimated theta with plug-in residual sigma; and
  (2) synthetic ORACLE truth with generation sigma.
The latter is validation-only and cannot be used on unknown experimental data.

F = J_log(theta)^T diag(n_k / sigma_k^2) J_log(theta),
C = F^(-1) via SVD, rho_ij = C_ij / sqrt(C_ii C_jj).

No thresholded FIM probability, combined GA+FIM+Bayesian reporting gate, or
GA+FIM reporting strategy is created anywhere by this script.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import zipfile
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dep_analysis_v5 import load_table, stratum_short

PARAMS=('eps_cyto','sigma_cyto','eps_shell','sigma_shell')
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]


def load_json(source,name):
    src=Path(source)
    if src.is_dir():
        p=src/name
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    with zipfile.ZipFile(src) as z:
        files=[p for p in z.namelist() if p==name or p.endswith('/'+name)]
        if not files:return {}
        if len(files)>1:raise ValueError('Ambiguous config JSON')
        return json.loads(z.read(files[0]).decode('utf-8'))


def svd_fim(jac,n_k,sigma_k):
    """Return full-covariance diagnostics, rejecting genuinely rank-deficient inputs.

    The rank check is intentionally weak (machine-precision based), to retain
    severely ill-conditioned S3 while flagging near-floating-point singularity.
    Do not interpret extremely large SD values as posterior uncertainty.
    """
    jac=np.asarray(jac,dtype=float)
    counts=np.asarray(n_k,dtype=float)
    sigma=np.broadcast_to(np.asarray(sigma_k,dtype=float),(len(counts),))
    if jac.shape!=(len(counts),4):raise ValueError('Jacobian must be K x 4')
    if np.any(~np.isfinite(jac)) or np.any(counts<=0) or np.any(sigma<=0):
        raise ValueError('Nonfinite Jacobian or invalid noise/counts')
    jw=jac*(np.sqrt(counts)/sigma)[:,None]
    u,s,vt=np.linalg.svd(jw,full_matrices=False)
    smin=s[-1];smax=s[0]
    ratio=smin/smax if smax>0 else 0.
    # Relative cutoff at machine precision, not customary 1e-8 that would
    # incorrectly mark S3 singular.
    singular=smin<=np.finfo(float).eps*smax*max(jw.shape)
    if singular:
        return dict(log10_cond=np.inf,log10_lmin=-np.inf,sd=np.full(4,np.nan),
                    corr=np.full((4,4),np.nan),singular=True)
    V=vt.T
    C=(V/s[None,:]) @ (V/s[None,:]).T
    C=(C+C.T)*.5
    sd=np.sqrt(np.clip(np.diag(C),0,np.inf))
    corr=C/np.outer(sd,sd)
    corr=np.clip(corr,-1,1)
    return dict(log10_cond=float(2*np.log10(smax/smin)),
                log10_lmin=float(2*np.log10(smin)),
                sd=sd,corr=corr,singular=False)


def jacobians(thetas):
    import jax.numpy as jnp
    import dep_synthetic_full_v5 as engine
    freq=engine.FREQ_6
    _,jbatch=engine.make_forward_and_jac(freq)
    out=[]
    for start in range(0,len(thetas),128):
        x=np.asarray(thetas[start:start+128],float)
        out.extend(np.asarray(jbatch(jnp.asarray(np.log(x))),float))
    return out


def diagnostic_rows(source,seed,recompute=True):
    ga_d=load_table(source,'results_GA_datasets.csv',True)
    ev=load_table(source,'evaluation_parameter_level.csv')
    truths=load_table(source,'selected_truths.csv')
    info=load_table(source,'FIM_condition_information_by_truth_noise.csv')
    widths=load_table(source,'FIM_local_widths_by_truth_and_noise.csv')
    config=load_json(source,'config.json')
    ga_d=ga_d[ga_d.ga_status.astype(str).eq('ok')].copy() if 'ga_status' in ga_d else ga_d.copy()
    if 'stratum' not in ga_d:raise ValueError('GA rows need stratum')
    ga_d['stratum']=ga_d.stratum.map(stratum_short)
    ga_d['seed_id']=seed
    ga_d['dataset_key']=seed+'::'+ga_d.dataset_id.astype(str)
    details=[];pairs=[]
    if not ev.empty and 'ga_fim_sd_ln' in ev:
        e=ev[ev.parameter.isin(PARAMS)][['dataset_id','parameter','ga_fim_sd_ln']].copy()
        e=e.merge(ga_d[['dataset_id','dataset_key','stratum','truth_id','noise_multiplier']],
                  on='dataset_id',how='inner',validate='many_to_one')
        e['seed_id']=seed
        e['location']='GA';e['source']='Saved V5 FIM local SD'
        for r in e.itertuples(index=False):
            details.append(dict(seed_id=seed,location='GA',source='saved_V5',dataset_key=r.dataset_key,
                truth_id=r.truth_id,stratum=r.stratum,noise_multiplier=r.noise_multiplier,
                parameter=r.parameter,sd_ln=float(r.ga_fim_sd_ln)))
    if not info.empty:
        info['seed_id']=seed
    if not widths.empty:
        widths['seed_id']=seed
        for r in widths.itertuples(index=False):
            if hasattr(r,'oracle_fim_sd_ln'):
                details.append(dict(seed_id=seed,location='TRUTH_ORACLE',source='saved_V5',
                    dataset_key=None,truth_id=r.truth_id,stratum=stratum_short(r.stratum),
                    noise_multiplier=r.noise_multiplier,parameter=r.parameter,
                    sd_ln=float(r.oracle_fim_sd_ln)))
    if recompute:
        try:
            import jax # noqa
            if not ga_d.empty:
                cols=['ga_best_'+p for p in PARAMS]
                if all(c in ga_d for c in cols):
                    theta=ga_d[cols].to_numpy(float)
                    js=jacobians(theta)
                    for r,jac in zip(ga_d.itertuples(index=False),js):
                        nk=float(getattr(r,'n_per_frequency',config.get('n_per_frequency',50)))
                        sig=float(getattr(r,'ga_sigma_mle',np.nan))
                        if not np.isfinite(sig) or sig<=0:continue
                        out=svd_fim(jac,np.repeat(nk,len(jac)),np.repeat(sig,len(jac)))
                        pkey=seed+'::'+str(r.dataset_id)
                        for i,prm in enumerate(PARAMS):
                            details.append(dict(seed_id=seed,location='GA',source='recomputed_full_FIM',
                                dataset_key=pkey,truth_id=r.truth_id,stratum=r.stratum,
                                noise_multiplier=r.noise_multiplier,parameter=prm,sd_ln=out['sd'][i],
                                log10_cond=out['log10_cond'],log10_lmin=out['log10_lmin'],
                                singular=out['singular']))
                        for i,k in PAIRS:
                            pairs.append(dict(seed_id=seed,location='GA',dataset_key=pkey,
                                truth_id=r.truth_id,stratum=r.stratum,noise_multiplier=r.noise_multiplier,
                                parameter_i=PARAMS[i],parameter_j=PARAMS[k],
                                FIM_corr=out['corr'][i,k],log10_cond=out['log10_cond'],singular=out['singular']))
            if not truths.empty and all(p in truths for p in PARAMS):
                theta=truths[list(PARAMS)].to_numpy(float)
                tmap={str(r.truth_id):j for r,j in zip(truths.itertuples(index=False),jacobians(theta))}
                # Oracle noise scenarios from V5 info; else use sigma_y reference.
                if not info.empty:
                    source_rows=info.drop_duplicates(['truth_id','noise_multiplier']).copy()
                else:
                    source_rows=truths[['truth_id','stratum']].copy()
                    source_rows['noise_multiplier']=1.
                    source_rows['sigma_true']=float(config.get('sigma_y',5))
                for r in source_rows.itertuples(index=False):
                    j=tmap.get(str(r.truth_id))
                    if j is None:continue
                    sig=float(getattr(r,'sigma_true',config.get('sigma_y',5.)))
                    nk=float(config.get('n_per_frequency',50))
                    if not np.isfinite(sig) or sig<=0:continue
                    out=svd_fim(j,np.repeat(nk,len(j)),np.repeat(sig,len(j)))
                    for i,prm in enumerate(PARAMS):
                        details.append(dict(seed_id=seed,location='TRUTH_ORACLE',source='recomputed_full_FIM',
                            dataset_key=None,truth_id=r.truth_id,stratum=stratum_short(r.stratum),
                            noise_multiplier=r.noise_multiplier,parameter=prm,sd_ln=out['sd'][i],
                            log10_cond=out['log10_cond'],log10_lmin=out['log10_lmin'],singular=out['singular']))
                    for i,k in PAIRS:
                        pairs.append(dict(seed_id=seed,location='TRUTH_ORACLE',dataset_key=None,
                            truth_id=r.truth_id,stratum=stratum_short(r.stratum),noise_multiplier=r.noise_multiplier,
                            parameter_i=PARAMS[i],parameter_j=PARAMS[k],FIM_corr=out['corr'][i,k],
                            log10_cond=out['log10_cond'],singular=out['singular']))
        except (ImportError,ModuleNotFoundError) as ex:
            warnings.warn(f'JAX/engine unavailable; only saved V5 FIM SD produced: {ex}')
    return details,pairs,ga_d,info,config


def plot_data(P,C,out):
    xorder=('S1','S2','S3','S4')
    if len(P):
        rows=P[(P.location=='GA')&(P.source=='recomputed_full_FIM')].copy()
        if rows.empty:rows=P[(P.location=='GA')&(P.source=='saved_V5')].copy()
        rows=rows[rows.sd_ln.gt(0)&np.isfinite(rows.sd_ln)]
        if not rows.empty:
            fig,ax=plt.subplots(figsize=(10,5.6))
            for prm in PARAMS:
                z=rows[rows.parameter==prm]
                y=[z[z.stratum==s].sd_ln.median() for s in xorder]
                ax.plot(list(xorder),y,'o-',lw=1.8,label=prm)
            ax.set_yscale('log');ax.set_ylabel('Median FIM local SD in ln(theta)')
            ax.set_xlabel('Synthetic ORACLE-truth FIM condition stratum')
            ax.set_title('GA-location FIM marginal uncertainty (diagnostic, NOT a gate)')
            ax.grid(alpha=.22);ax.legend()
            fig.tight_layout();fig.savefig(out/'01_FIM_SD_by_stratum.png',dpi=190);plt.close(fig)
    if len(C):
        rows=C[(C.location=='TRUTH_ORACLE')&np.isfinite(C.FIM_corr)].copy()
        if rows.empty:rows=C[(C.location=='GA')&np.isfinite(C.FIM_corr)].copy()
        if not rows.empty:
            grp=rows.groupby(['seed_id','location','truth_id','noise_multiplier','stratum'],dropna=False).FIM_corr.agg(lambda s:np.max(np.abs(s))).reset_index(name='max_abs_corr')
            fig,ax=plt.subplots(figsize=(8.5,5))
            x=[];y=[]
            for i,s in enumerate(xorder):
                z=grp[grp.stratum==s]
                if z.empty:continue
                xs=np.full(len(z),i,dtype=float)+np.linspace(-.07,.07,len(z))
                ax.scatter(xs,z.max_abs_corr,alpha=.45,s=18)
                ax.hlines(z.max_abs_corr.median(),i-.20,i+.20,colors='black',lw=2.4)
            ax.axhline(.8,ls='--',color='gray',lw=1)
            ax.set(xticks=range(4),xticklabels=xorder,ylabel='Maximum |FIM parameter correlation|',
                   ylim=(-.03,1.04),title='Local parameter compensation (diagnostic, NOT a gate)')
            ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(out/'02_FIM_correlation_by_stratum.png',dpi=190);plt.close(fig)



def posterior_correlation_diagnostics(sources,seed_labels,out):
    """Context only: posterior correlations are NOT FIM correlations or gates."""
    rows=[];widths=[]
    for source,seed in zip(sources,seed_labels):
        d=load_table(source,'results_datasets.csv')
        if not d.empty and 'mcmc_pass' in d:
            d=d[d.mcmc_pass.astype(str).str.lower().eq('true')].copy()
            if 'status' in d:d=d[d.status.astype(str).eq('ok')]
            corr_cols=[c for c in d if c.startswith('posterior_corr_ln_')]
            for r in d.itertuples(index=False):
                vals=[abs(float(getattr(r,c))) for c in corr_cols
                      if pd.notna(getattr(r,c)) and np.isfinite(float(getattr(r,c)))]
                if vals:
                    rows.append(dict(seed_id=seed,dataset_id=r.dataset_id,
                        stratum=stratum_short(r.stratum),max_abs_posterior_corr=max(vals),
                        n_pairs_available=len(vals)))
        e=load_table(source,'evaluation_parameter_level.csv')
        if not e.empty and 'posterior_ci_width_percent_prior' in e:
            if 'mcmc_pass' in e:e=e[e.mcmc_pass.astype(str).str.lower().eq('true')]
            for r in e[e.parameter.isin(PARAMS)].itertuples(index=False):
                widths.append(dict(seed_id=seed,dataset_id=r.dataset_id,
                    stratum=stratum_short(r.stratum),parameter=r.parameter,
                    posterior_ci_width_percent_prior=float(r.posterior_ci_width_percent_prior)))
    posterior=pd.DataFrame(rows)
    if not posterior.empty:
        posterior.to_csv(out/'POSTERIOR_max_correlation_context.csv',index=False)
        summary=posterior.groupby('stratum').agg(
          n=('max_abs_posterior_corr','size'),
          median_max_abs_posterior_corr=('max_abs_posterior_corr','median'),
          fraction_above_08=('max_abs_posterior_corr',lambda z:float((z>.8).mean()))).reset_index()
        summary.to_csv(out/'POSTERIOR_correlation_context_summary.csv',index=False)
    widths=pd.DataFrame(widths)
    if not widths.empty:
        widths.to_csv(out/'POSTERIOR_width_context.csv',index=False)
        widths.groupby(['stratum','parameter']).posterior_ci_width_percent_prior.median().reset_index(
            name='median_width_percent_prior').to_csv(out/'POSTERIOR_width_context_summary.csv',index=False)
    return posterior


def compare_corr_plot(C,posterior,out):
    if C.empty or posterior.empty:return
    F=C[(C.location=='TRUTH_ORACLE')&np.isfinite(C.FIM_corr)].copy()
    if F.empty:return
    g=F.groupby(['seed_id','truth_id','stratum','noise_multiplier'],dropna=False).FIM_corr.agg(
        lambda x:np.max(np.abs(x))).reset_index(name='max_abs_FIM_corr')
    fig,ax=plt.subplots(figsize=(8.5,5))
    for title,series in [('FIM at synthetic truth',g.groupby('stratum').max_abs_FIM_corr.median()),
                         ('NUTS posterior',posterior.groupby('stratum').max_abs_posterior_corr.median())]:
        y=[series.get(s,np.nan) for s in ('S1','S2','S3','S4')]
        ax.plot(['S1','S2','S3','S4'],y,'o-',label=title)
    ax.set(ylabel='Median of maximum absolute pairwise correlation',ylim=(0,1.04),
           title='FIM and posterior correlation (different approximations / samples)',
           xlabel='Stratum at synthetic oracle truth')
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'03_FIM_vs_posterior_correlation_context.png',dpi=190);plt.close(fig)


def run_fim(sources,out,seed_labels=None,recompute=True):
    out=Path(out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    if seed_labels is None:seed_labels=[f'seed{i}' for i in range(len(sources))]
    if len(seed_labels)!=len(sources):raise ValueError('seed labels count mismatch')
    details=[];pairs=[];contexts=[]
    for source,seed in zip(sources,seed_labels):
        d,c,g,info,config=diagnostic_rows(source,seed,recompute)
        details.extend(d);pairs.extend(c)
        if not info.empty:
            contexts.append(info)
        elif not g.empty:
            contexts.append(g[['dataset_key','truth_id','stratum','noise_multiplier','log10_fim_cond']].copy())
    P=pd.DataFrame(details);C=pd.DataFrame(pairs)
    if P.empty:raise ValueError('No FIM diagnostics found. Need V5 evaluation files or JAX engine to recompute.')
    if not C.empty:
        C['pair']=C.parameter_i+'__'+C.parameter_j
        C['abs_corr']=C.FIM_corr.abs()
    P.to_csv(out/'FIM_parameter_uncertainty.csv',index=False)
    if not C.empty:C.to_csv(out/'FIM_pairwise_correlations.csv',index=False)
    if contexts:pd.concat(contexts,ignore_index=True).to_csv(out/'FIM_information_context.csv',index=False)
    summary=[]
    for (loc,source,st,prm),z in P.groupby(['location','source','stratum','parameter']):
        valid=z.sd_ln[np.isfinite(z.sd_ln)]
        summary.append(dict(location=loc,source=source,stratum=st,parameter=prm,
            n=len(z),n_finite=len(valid),median_sd_ln=float(valid.median()) if len(valid) else np.nan,
            q25_sd_ln=float(valid.quantile(.25)) if len(valid) else np.nan,
            q75_sd_ln=float(valid.quantile(.75)) if len(valid) else np.nan))
    pd.DataFrame(summary).to_csv(out/'FIM_SD_stratum_summary.csv',index=False)
    if not C.empty:
        gm=C.groupby(['seed_id','location','stratum','dataset_key','truth_id','noise_multiplier'],dropna=False).agg(
            max_abs_corr=('abs_corr','max')).reset_index()
        gm.to_csv(out/'FIM_per_case_max_correlation.csv',index=False)
        rr=gm.groupby(['location','stratum']).agg(n_cases=('max_abs_corr','size'),
            median_max_abs_corr=('max_abs_corr','median'),
            fraction_max_corr_above_08=('max_abs_corr',lambda s:float((s>.8).mean()))).reset_index()
        rr.to_csv(out/'FIM_correlation_stratum_summary.csv',index=False)
    plot_data(P,C,out)
    posterior_context=posterior_correlation_diagnostics(sources,seed_labels,out)
    compare_corr_plot(C,posterior_context,out)
    meta=dict(sources=list(map(str,sources)),n_parameter_rows=len(P),n_pairwise_rows=len(C),
        strata='S1-S4 classified by ORACLE FIM condition number at synthetic truth',
        GA_FIM='GA-location plug-in noise and actual synthetic observation count',
        oracle_FIM='Truth-location for post-hoc characterization ONLY; not available in real measurements',
        reporting='NONE. FIM is never a hard gate and no GA+FIM+Bayesian decisions are created',
        diagnostics='Unbounded local Gaussian approximation; extreme SD values may signal rank deficiency or curved likelihood. Correlations describe local linear compensation, not global posterior correlation.',
        recomputed_full_FIM=recompute)
    (out/'FIM_METADATA.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
    print('[FIM diagnostic ONLY]',len(P),'parameter rows',len(C),'pair rows; saved',out)
    return P,C


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources',nargs='+',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--seed-labels',nargs='+',default=None)
    ap.add_argument('--no-recompute-correlations',action='store_true',help='Use saved V5 FIM SD only, no JAX')
    a=ap.parse_args()
    run_fim(a.sources,a.output_dir,a.seed_labels,not a.no_recompute_correlations)

if __name__=='__main__':main()
