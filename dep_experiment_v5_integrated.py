#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrated V5: raw DEP data -> GA -> feasible NUTS -> GA/Bayesian reporting; independent FIM diagnostics.

Original experimental XLSX layout:
   row 1: numeric frequency headers (Hz), exactly [0.5,1,2,5,10,50] MHz
   rows 2+: raw DEP velocity values (um/s), blanks allowed independently per column
Alternative long CSV: frequency_hz,velocity_um_s

The program uses ALL raw individual velocities with actual counts n_k. It NEVER
pads missing cells, assumes n=50, or treats column index as matched replicates.
Default is the same constant-sigma Gaussian likelihood as the synthetic paper.
--noise-model per_frequency is an optional sensitivity analysis, NOT directly
comparable to the common-noise synthetic benchmark unless rebenchmarked.

Real experimental data have NO truth dielectric properties: do NOT report
empirical parameter recovery, false acceptance rates or posterior coverage.
GA-centered probability means model-conditional POSTERIOR MASS, not a validated
probability the GA estimate equals a physical property.
"""
from __future__ import annotations
import argparse
import json
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dep_synthetic_full_v5 as engine
from dep_synthetic_full_v5 import (PARAM_NAMES,LOG_NAMES,LOW,HIGH,LN_LOW,LN_HIGH,
    FREQ_6,EPS0,PRIOR_SPAN_LOG10,one_ga_start,ga_velocity_numpy,
    make_forward_and_jac,load_control_observations,fim_gaussian_recovery_probability)

GATES=(.5,.6,.7,.8)


def args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--input',type=Path,required=True,help='Original .xlsx or long-format .csv')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--sheet',default=None)
    p.add_argument('--dataset-id',default=None)
    p.add_argument('--noise-model',choices=['common','per_frequency'],default='common')
    # The benchmark's original physical constants must be checked against real hardware.
    p.add_argument('--electric-field-v-m',type=float,default=30000.,
                   help='Actual electric-field scale; benchmark default 25000 V/m')
    p.add_argument('--field-gradient-length-m',type=float,default=9e-6,
                   help='Benchmark approximation E^2 / 9e-6 m; replace with FEM-calibrated gradient')
    p.add_argument('--radius-m',type=float,default=3.055e-6)
    p.add_argument('--shell-thickness-m',type=float,default=55e-9)
    p.add_argument('--medium-conductivity-s-m',type=float,default=.1)
    p.add_argument('--medium-permittivity-relative',type=float,default=80.)
    p.add_argument('--viscosity-pa-s',type=float,default=1e-3)
    p.add_argument('--ga-only',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--ga-runs',type=int,default=50)
    p.add_argument('--ga-population',type=int,default=100)
    p.add_argument('--ga-generations',type=int,default=500)
    p.add_argument('--ga-elite-fraction',type=float,default=.10)
    p.add_argument('--ga-tournament-size',type=int,default=3)
    p.add_argument('--ga-crossover',type=float,default=.80)
    p.add_argument('--ga-mutation',type=float,default=.30)
    p.add_argument('--ga-log-mutation-sd',type=float,default=.35)
    p.add_argument('--ga-early-stop',type=int,default=60)
    p.add_argument('--ga-local-polish',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--chains',type=int,default=5)
    p.add_argument('--chain-method',choices=['sequential','parallel','vectorized'],default='sequential')
    p.add_argument('--warmup',type=int,default=15000)
    p.add_argument('--samples',type=int,default=35000)
    p.add_argument('--target-accept',type=float,default=.99)
    p.add_argument('--max-tree-depth',type=int,default=10)
    p.add_argument('--max-rhat',type=float,default=1.01)
    p.add_argument('--min-ess',type=float,default=1000)
    p.add_argument('--dense-mass',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--init-strategy',choices=['feasible','uniform'],default='feasible',
                   help='Default feasible; uniform uses random unconstrained [-radius,+radius]')
    p.add_argument('--init-radius',type=float,default=2.,
                   help='Only used when --init-strategy uniform; feasible sets radius=0')
    p.add_argument('--ppc-draws',type=int,default=500)
    p.add_argument('--save-posteriors',action='store_true')
    p.add_argument('--save-idata',action='store_true',help='Save ArviZ InferenceData NetCDF')
    p.add_argument('--ppc-quantile-method',choices=['analytic','montecarlo'],default='analytic',
                   help='Analytic mixture-CDF quantiles (smooth) or Monte Carlo noise draws')
    p.add_argument('--ppc-noise-draws',type=int,default=5,
                   help='Predictive noise replicates per selected posterior sample')
    p.add_argument('--dense-points',type=int,default=500)
    p.add_argument('--corner-max-draws',type=int,default=2500)
    p.add_argument('--marginal-style',choices=['kde','hist+kde'],default='kde',
                   help='Smooth posterior marginal densities, optionally with a faint density histogram')
    p.add_argument('--marginal-kde-bandwidth',type=float,default=1.0,
                   help="Gaussian KDE bandwidth factor relative to Scott's rule; 1.0 is default")
    p.add_argument('--marginal-max-draws',type=int,default=8000,
                   help='Maximum posterior samples used per KDE visualization (not inference)')
    p.add_argument('--map-candidates',type=int,default=5000,
                   help='Approximate MAP searches only this many posterior draws')
    p.add_argument('--plot-jitter-log10',type=float,default=0.0,
                   help='Visualization-only horizontal jitter in log10 MHz; raw frequencies unchanged')
    return p.parse_args()




def configure_physics(a):
    inputs=[a.electric_field_v_m,a.field_gradient_length_m,a.radius_m,
            a.shell_thickness_m,a.medium_conductivity_s_m,
            a.medium_permittivity_relative,a.viscosity_pa_s]
    if not all(np.isfinite(v) and v>0 for v in inputs):
        raise ValueError('All forward-model physical constants must be positive, finite values')
    if a.shell_thickness_m>=a.radius_m:raise ValueError('Shell thickness must be smaller than radius')
    engine.E_FIELD=float(a.electric_field_v_m)
    engine.RADIUS=float(a.radius_m)
    engine.T_SHELL=float(a.shell_thickness_m)
    engine.EPS_MED=float(a.medium_permittivity_relative)*EPS0
    engine.SIG_MED=float(a.medium_conductivity_s_m)
    engine.VISCOSITY=float(a.viscosity_pa_s)
    engine.DEP_CONST=((engine.RADIUS**2*engine.EPS_MED*
       (engine.E_FIELD**2/a.field_gradient_length_m))/(3*engine.VISCOSITY))*1e6
    return dict(electric_field_v_m=engine.E_FIELD,field_gradient_length_m=a.field_gradient_length_m,
        radius_m=engine.RADIUS,shell_thickness_m=engine.T_SHELL,
        medium_conductivity_s_m=engine.SIG_MED,
        medium_permittivity_relative=a.medium_permittivity_relative,
        viscosity_pa_s=engine.VISCOSITY,DEP_CONST=engine.DEP_CONST,
        prior_parameter_order=list(PARAM_NAMES),
        prior_lower=LOW.tolist(),prior_upper=HIGH.tolist(),
        interpretation='Single-shell model; electric-field prefactor must be matched to hardware.')

def load_raw(path,sheet):
    # Reuses the validated reader from the 2026-10-31 research code.
    df=load_control_observations(path,sheet_name=sheet)
    freqs=pd.to_numeric(df.frequency_hz,errors='coerce').to_numpy(float)
    values=pd.to_numeric(df.velocity_um_s,errors='coerce').to_numpy(float)
    if not (np.all(np.isfinite(freqs)) and np.all(np.isfinite(values))):
        raise ValueError('Nonfinite frequency/velocity in file')
    match=np.isclose(freqs[:,None],FREQ_6[None,:],rtol=1e-6,atol=1e-3)
    if np.any(match.sum(axis=1)!=1):
        raise ValueError('Only exact 0.5,1,2,5,10,50 MHz frequencies are supported. '
                         'Modify physical design and forward grid explicitly for alternatives.')
    idx=match.argmax(axis=1)
    raw=[values[idx==i].astype(float) for i in range(len(FREQ_6))]
    count=np.array([len(v) for v in raw],int)
    if np.any(count<1):raise ValueError('Each of the six frequencies must contain >=1 raw velocity')
    long=pd.DataFrame({'frequency_hz':np.repeat(FREQ_6,count),
                       'velocity_um_s':np.concatenate(raw),
                       'frequency_index':np.repeat(np.arange(6),count)})
    return raw,count,long


def weight_fim(jac,counts,sigma_freq):
    # F in phi=ln(theta): sum_k n_k/sigma_k^2 * J_k^T J_k.
    # Independent observation Gaussian working model.
    jw=np.asarray(jac,float) * (np.sqrt(counts)/sigma_freq)[:,None]
    _,sv,vt=np.linalg.svd(jw,full_matrices=False)
    good=sv>0
    if not np.all(good):
        sd=np.full(4,np.inf)
    else:
        v=vt.T
        sd=np.sqrt(np.sum((v / sv[None,:])**2,axis=1))
    log_cond=float(2*np.log10(sv[0]/sv[-1])) if sv[-1]>0 else float('inf')
    return dict(sd_ln=sd,log10_fim_cond=log_cond,
        log10_lambda_min=float(2*np.log10(sv[-1])) if sv[-1]>0 else -np.inf,
        F=jw.T@jw,sv=sv)


def posterior_model(freq,which,noise_model):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from dep_synthetic_full_v5 import single_velocity
    f=jnp.asarray(freq,dtype=jnp.float64)
    group=jnp.asarray(which,dtype=jnp.int32)
    lower=jnp.asarray(LN_LOW);upper=jnp.asarray(LN_HIGH)
    def model(y):
        logt=jnp.stack([numpyro.sample(n,dist.Uniform(lower[i],upper[i]))
                        for i,n in enumerate(LOG_NAMES)])
        theta=jnp.exp(logt)
        mu=single_velocity(f,theta)
        if noise_model=='common':
            sigma=numpyro.sample('sigma_y',dist.HalfCauchy(5.))
            loc=mu[group];scale=sigma
        else:
            sigma=numpyro.sample('sigma_by_frequency',dist.HalfCauchy(5.).expand([len(f)]).to_event(1))
            loc=mu[group];scale=sigma[group]
        numpyro.sample('observed_velocities',dist.Normal(loc,scale).to_event(1),obs=y)
    return model


def plot_ga(raw,means,sd,ga_theta,out,did):
    dense=np.geomspace(min(FREQ_6),max(FREQ_6),240)
    fig,ax=plt.subplots(figsize=(9,5.5))
    for i,v in enumerate(raw):
        ax.scatter(np.full(len(v),FREQ_6[i]/1e6),v,s=9,alpha=.16,color='#547c97')
    ax.errorbar(FREQ_6/1e6,means,yerr=np.nan_to_num(sd/np.sqrt([len(v) for v in raw])),
        fmt='o',capsize=3,color='black',label='Observed mean +/- SE')
    ax.plot(dense/1e6,ga_velocity_numpy(dense,np.log(ga_theta)),color='#D97706',lw=2,
            label='Best multistart GA fit')
    ax.set_xscale('log');ax.set_xlabel('Frequency (MHz)');ax.set_ylabel('Velocity (um/s)')
    ax.set_title(f'{did} : actual irregular-n raw velocity + GA');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'GA_fitting.png',dpi=170);plt.close(fig)


def diagnostic_from_mcmc(mcmc,a):
    import arviz as az
    summary=az.summary(az.from_numpyro(mcmc),var_names=list(LOG_NAMES)+
                       (['sigma_y'] if a.noise_model=='common' else ['sigma_by_frequency']))
    summary.to_csv(a.output_dir/'nuts_arviz_summary.csv')
    r=summary['r_hat'].to_numpy(float);eb=summary['ess_bulk'].to_numpy(float)
    et=summary['ess_tail'].to_numpy(float)
    extras=mcmc.get_extra_fields()
    div=int(np.asarray(extras['diverging']).sum())
    passed=bool(div==0 and np.all(np.isfinite(r)) and np.max(r)<a.max_rhat
                and np.all(np.isfinite(eb)) and np.min(eb)>=a.min_ess
                and np.all(np.isfinite(et)) and np.min(et)>=a.min_ess)
    return dict(nuts_pass=passed,divergences=div,max_rhat=float(np.nanmax(r)),
        min_ess_bulk=float(np.nanmin(eb)),min_ess_tail=float(np.nanmin(et)),
        num_chains=a.chains,warmup=a.warmup,samples_per_chain=a.samples,
        target_accept=a.target_accept,noise_model=a.noise_model)


def decisions(tab,diag,out):
    """GA-only and four GA+Bayesian reporting rules. FIM is NEVER a gate."""
    keep=['parameter','GA_estimate','posterior_prob_GA_factor2']
    keep=[c for c in keep if c in tab.columns]
    t=tab[keep].copy()
    t['GA-only']=True
    not_run=not diag.get('nuts_run',True)
    for gate in GATES:
        col=f'GA+Bayes_{gate:.1f}'
        if not_run:
            t[col]=pd.Series(pd.NA,index=t.index,dtype='boolean')
        else:
            t[col]=bool(diag['nuts_pass'])&t.posterior_prob_GA_factor2.ge(gate)
    t['nuts_stage']=('NOT_RUN' if not_run else 'DIAGNOSTIC_PASS' if diag['nuts_pass'] else 'DIAGNOSTIC_FAIL')
    t.to_csv(out/'parameter_reporting_gates.csv',index=False)
    rows=[]
    for m in ['GA-only']+[f'GA+Bayes_{g:.1f}' for g in GATES]:
        eligible=not (not_run and 'Bayes' in m)
        count=int(t[m].sum()) if eligible else None
        rows.append(dict(method=m,n_parameters_accepted=count,
            all_four_accepted=(count==4) if eligible else pd.NA,
            at_least_three_accepted=(count>=3) if eligible else pd.NA,
            status='ASSESSED' if eligible else 'NOT_RUN',
            note='Experimental truth unknown; posterior mass is model-conditional. FIM is diagnostic-only.'))
    pd.DataFrame(rows).to_csv(out/'dataset_reporting_gates.csv',index=False)
    return t


def main():
    a=args();a.output_dir=a.output_dir.expanduser().resolve();out=a.output_dir
    if not np.isfinite(a.init_radius) or a.init_radius<0:
        raise ValueError('--init-radius must be finite and >=0')
    if a.corner_max_draws<1:
        raise ValueError('--corner-max-draws must be >=1')
    out.mkdir(parents=True,exist_ok=True)
    physical=configure_physics(a)
    physical['sampler_initialization']=a.init_strategy
    physical['sampler_uniform_radius']=a.init_radius if a.init_strategy=='uniform' else 0.0
    physical['sampler_dense_mass']=a.dense_mass
    (out/'PHYSICAL_MODEL_CONFIG.json').write_text(json.dumps(physical,indent=2),encoding='utf8')
    print('[MODEL] Experimental constants from CLI (verify vs your electrode geometry):',physical)
    raw,counts,L=load_raw(a.input,a.sheet)
    did=a.dataset_id or a.input.stem
    L.to_csv(out/'raw_velocities_long.csv',index=False)
    means=np.array([np.mean(v) for v in raw]);sd=np.array([np.std(v,ddof=1) if len(v)>1 else np.nan for v in raw])
    observed_se=sd/np.sqrt(counts)
    base_sigma=np.sqrt(sum(np.sum((v-means[i])**2) for i,v in enumerate(raw))/max(1,len(L)))
    if a.noise_model=='per_frequency':
        if np.any(counts<2):
            warnings.warn('Some columns have only one observation; pooled sigma fallback used for GA weights.')
        estimated_sigma=np.where(np.isfinite(sd)&(sd>0),sd,max(base_sigma,1e-6))
        # Weighted Gaussian least squares to match the per-frequency error model.
        ga_weights=counts/estimated_sigma**2
    else:
        estimated_sigma=np.full(6,base_sigma)
        ga_weights=counts.astype(float)
    fs=pd.DataFrame({'frequency_hz':FREQ_6,'n_raw':counts,'velocity_mean_um_s':means,
        'velocity_sd_um_s':sd,'velocity_se_um_s':observed_se,
        'working_noise_sigma_um_s':estimated_sigma,'ga_weight':ga_weights})
    fs.to_csv(out/'frequency_summary.csv',index=False)
    print('[EXPERIMENT] Measured n_k:',dict(zip(FREQ_6.astype(int),counts)),'total=',counts.sum())
    # Warn when using an assumption inconsistent with observed variances.
    sd_good=sd[np.isfinite(sd)&(sd>0)]
    if len(sd_good)>1 and sd_good.max()/sd_good.min()>2.0 and a.noise_model=='common':
        warnings.warn('Observed SD differs >2x across frequencies; common-noise model may be misspecified. '
                      'Run --noise-model per_frequency as sensitivity analysis.')
    rng=np.random.default_rng(a.seed)
    runs=[]
    for i in range(a.ga_runs):
        result=one_ga_start(a,FREQ_6,means,int(rng.integers(0,2**31-1)),n_per_freq=ga_weights)
        result['run_id']=i+1
        runs.append(result)
    trials=pd.DataFrame(runs)
    trials.to_csv(out/'GA_multistart_runs.csv',index=False)
    good=trials[np.isfinite(trials.ga_sse_mean_final)]
    if good.empty:raise RuntimeError('GA had no finite result; check data and physical model')
    best=good.loc[good.ga_sse_mean_final.idxmin()]
    ga_theta=np.array([float(best['ga_'+p]) for p in PARAM_NAMES])
    ga_pred=ga_velocity_numpy(FREQ_6,np.log(ga_theta))
    residuals=np.concatenate([v-ga_pred[i] for i,v in enumerate(raw)])
    sigma_mle=float(np.sqrt(np.mean(residuals**2)))
    # FIM MUST use actual n_k; optional per-frequency plug-in variances.
    if a.noise_model=='common':
        noise_for_fim=np.full(6,max(sigma_mle,1e-8))
    else:
        noise_for_fim=np.array([max(float(np.sqrt(np.mean((v-ga_pred[i])**2))),1e-8)
                      for i,v in enumerate(raw)])
    _,jacbatch=make_forward_and_jac(FREQ_6)
    import jax.numpy as jnp
    jac=np.asarray(jacbatch(jnp.asarray(np.log(ga_theta)[None,:])),float)[0]
    fim=weight_fim(jac,counts,noise_for_fim)
    fi_prob=fim_gaussian_recovery_probability(fim['sd_ln'],2.)
    ga_param=pd.DataFrame([dict(parameter=p,GA_estimate=ga_theta[i],
        GA_relative_permittivity=(ga_theta[i]/EPS0 if p.startswith('eps_') else np.nan),
        ga_fim_sd_ln=fim['sd_ln'][i],ga_fim_prob_factor2=float(fi_prob[i]),
        ga_fim_log10_95_halfwidth=1.95996398454*fim['sd_ln'][i]/math.log(10),
        note='GA-centered conditional local Gaussian FIM approximation')
        for i,p in enumerate(PARAM_NAMES)])
    # Keep the legacy CSV solely for KDE replot compatibility. It is DIAGNOSTIC,
    # not an active reporting gate. Dedicated FIM tables are also saved below.
    ga_param.to_csv(out/'GA_FIM_parameters.csv',index=False)
    ga_param[['parameter','GA_estimate','GA_relative_permittivity']].to_csv(
        out/'GA_parameters.csv',index=False)
    ga_param[['parameter','ga_fim_sd_ln','ga_fim_prob_factor2',
              'ga_fim_log10_95_halfwidth','note']].to_csv(
        out/'FIM_diagnostics_parameters.csv',index=False)
    pd.DataFrame(fim['F'],index=PARAM_NAMES,columns=PARAM_NAMES).to_csv(
        out/'FIM_matrix_ln.csv')
    # Build the covariance using the same SVD geometry used to calculate sd_ln.
    U,sv,vt=np.linalg.svd((jac*np.sqrt(counts)[:,None]/noise_for_fim[:,None]),
                            full_matrices=False)
    if np.all(sv>np.finfo(float).eps*sv[0]*max(jac.shape)):
        V=vt.T
        C=(V/sv[None,:])@(V/sv[None,:]).T
        Corr=C/np.sqrt(np.outer(np.diag(C),np.diag(C)))
        pd.DataFrame(C,index=PARAM_NAMES,columns=PARAM_NAMES).to_csv(
            out/'FIM_covariance_ln.csv')
        pd.DataFrame(Corr,index=PARAM_NAMES,columns=PARAM_NAMES).to_csv(
            out/'FIM_correlation_ln.csv')
    summary=dict(dataset_id=did,noise_model=a.noise_model,n_total=int(counts.sum()),
        n_per_frequency=counts.tolist(),ga_mle_sigma_um_s=sigma_mle,
        ga_minimized_weighted_residual=float(best.ga_sse_mean_final),
        log10_ga_fim_cond=fim['log10_fim_cond'],log10_ga_fim_lmin=fim['log10_lambda_min'],
        ga_optimizations=a.ga_runs,ga_best_run=int(best.run_id))
    (out/'GA_FIM_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    (out/'FIM_diagnostics_summary.json').write_text(json.dumps(
        {**summary,'role':'IDENTIFIABILITY DIAGNOSTICS ONLY. No FIM reporting gate.'},
        indent=2),encoding='utf8')
    plot_ga(raw,means,sd,ga_theta,out,did)
    if a.ga_only:
        prelim=ga_param.rename(columns={'ga_fim_prob_factor2':'ga_fim_prob_factor2'}).copy()
        prelim['posterior_prob_GA_factor2']=np.nan
        decisions(prelim,dict(nuts_pass=False,nuts_run=False),out)
        print('[GA-only] saved GA-only reporting + INDEPENDENT FIM diagnostics. Bayesian probabilities are UNKNOWN.')
        print('[GA-only] NUTS, empirical parameter coverage, and calibration NOT computed.')
        return
    try:
        import jax
        import jax.numpy as jnp
        from numpyro.infer import MCMC,NUTS
        from numpyro.infer.initialization import init_to_feasible, init_to_uniform
    except ImportError as e:
        raise SystemExit('Install NumPyro, JAX and ArviZ to run NUTS. GA-only outputs are already saved. '+str(e))
    if a.chains>jax.local_device_count() and a.chain_method=='parallel':
        raise ValueError('Parallel chains require enough devices; use sequential')
    model=posterior_model(FREQ_6,L.frequency_index.to_numpy(int),a.noise_model)
    init_strategy=(init_to_feasible() if a.init_strategy=='feasible'
                   else init_to_uniform(radius=a.init_radius))
    print('[SAMPLER] initialization=',a.init_strategy,'radius=',
          0. if a.init_strategy=='feasible' else a.init_radius,
          'dense_mass=',a.dense_mass)
    sampler=MCMC(NUTS(model,target_accept_prob=a.target_accept,max_tree_depth=a.max_tree_depth,
        dense_mass=a.dense_mass,init_strategy=init_strategy),num_warmup=a.warmup,num_samples=a.samples,
        num_chains=a.chains,chain_method=a.chain_method,progress_bar=True)
    sampler.run(jax.random.PRNGKey(a.seed+13429),y=jnp.asarray(L.velocity_um_s.to_numpy(float)),
        extra_fields=('diverging','num_steps','accept_prob','energy'))
    diag=diagnostic_from_mcmc(sampler,a)
    samples={k:np.asarray(v) for k,v in sampler.get_samples().items()}
    if a.save_posteriors:np.savez_compressed(out/'experiment_joint_posterior.npz',**samples)
    if a.save_idata:
        import arviz as az
        az.from_numpyro(sampler).to_netcdf(out/'experiment_idata.nc')
    m=np.column_stack([samples['log_'+p].reshape(-1) for p in PARAM_NAMES])
    corr=np.corrcoef(m,rowvar=False)
    pd.DataFrame(corr,index=PARAM_NAMES,columns=PARAM_NAMES).to_csv(out/'posterior_correlation_ln.csv')
    # Marginal posterior quantities are computed from true aligned draws.
    parameter_rows=[]
    for i,p in enumerate(PARAM_NAMES):
        dd=m[:,i]
        a025,amed,a975=np.quantile(dd,[.025,.5,.975]);g=float(np.log(ga_theta[i]));radius=math.log(2)
        parameter_rows.append(dict(parameter=p,GA_estimate=ga_theta[i],
            GA_relative_permittivity=ga_theta[i]/EPS0 if p.startswith('eps_') else np.nan,
            posterior_median=float(np.exp(amed)),posterior_ci95_low=float(np.exp(a025)),
            posterior_ci95_high=float(np.exp(a975)),
            posterior_relative_permittivity_median=float(np.exp(amed)/EPS0) if p.startswith('eps_') else np.nan,
            posterior_width_log10=float((a975-a025)/math.log(10)),
            posterior_width_percent_prior=float(100*(a975-a025)/math.log(HIGH[i]/LOW[i])),
            posterior_prob_GA_factor2=float(np.mean(np.abs(dd-g)<=radius)),
            posterior_prob_Bayes_median_factor2=float(np.mean(np.abs(dd-amed)<=radius)),
            max_abs_posterior_corr_to_other=float(np.max(np.abs(np.delete(corr[i],i)))),
            NUTS_diagnostics_pass=diag['nuts_pass']))
    param=pd.DataFrame(parameter_rows)
    param.to_csv(out/'fit_parameters_posterior.csv',index=False)
    accepted=decisions(param,diag,out)
    hits=np.abs(m-np.log(ga_theta)[None,:])<=math.log(2)
    joint=float(np.mean(np.all(hits,axis=1)))
    atleast3=float(np.mean(hits.sum(axis=1)>=3))
    joint_row=dict(**diag,posterior_joint_prob_4of4_GA_factor2=joint,
        posterior_prob_atleast3_GA_factor2=atleast3,
        product_marginal_prob=float(np.prod(param.posterior_prob_GA_factor2)),
        note='Model-conditional support ONLY; empirical coverage cannot be assessed without truth')
    (out/'NUTS_diagnostics_and_joint.json').write_text(json.dumps(joint_row,indent=2),encoding='utf8')
    # PPC at OBSERVED frequencies from correlated draws and sigma posterior.
    pick=np.random.default_rng(a.seed+70).choice(len(m),size=min(a.ppc_draws,len(m)),replace=False)
    vel_preds=ga_velocity_numpy(FREQ_6,m[pick])
    sig=np.asarray(samples['sigma_y'])[pick,None] if a.noise_model=='common' else np.asarray(samples['sigma_by_frequency'])[pick]
    # Compute the V5-compatible PPC CSV using the SAME predictive quantiles
    # as the extended coverage/figure pipeline (prevents inconsistent outputs).
    from dep_experiment_v5_extended import gaussian_mixture_quantiles, _predictive_sample
    if a.ppc_quantile_method=='analytic':
        qi=gaussian_mixture_quantiles(vel_preds,sig,reference_n=1)
        qm=gaussian_mixture_quantiles(vel_preds,sig,reference_n=counts)
        lo,hi=qi[0],qi[4];mlo,mhi=qm[0],qm[4]
    else:
        replicated,mrep=_predictive_sample(vel_preds,sig,counts,a.ppc_noise_draws,a.seed+301)
        lo,hi=np.quantile(replicated,[.025,.975],axis=0)
        mlo,mhi=np.quantile(mrep,[.025,.975],axis=0)
    pred_summ=pd.DataFrame(dict(frequency_hz=FREQ_6,n_raw=counts,observed_mean=means,
        observed_sd=sd,observed_SE=observed_se,GA_velocity=ga_pred,
        Bayes_median_velocity=ga_velocity_numpy(FREQ_6,np.log(param.posterior_median.to_numpy(float))),
        ppc_individual_low=lo,ppc_individual_high=hi,
        ppc_mean_low=mlo,ppc_mean_high=mhi,
        observed_individual_fraction_in_PPC95=np.array([np.mean((v>=lo[i])&(v<=hi[i])) for i,v in enumerate(raw)]),
        observed_mean_in_PPC95=(means>=mlo)&(means<=mhi)))
    pred_summ.to_csv(out/'PPC_in_sample_by_frequency.csv',index=False)
    # Extended KDE, PPC and posterior plots do not use FIM reporting gates.
    from dep_experiment_v5_extended import make_extended_outputs
    make_extended_outputs(a,out,raw,counts,means,sd,ga_theta,samples,diag,sampler)
    # Backward-compatible old V5 figure filename, now WITH all raw observations.
    import shutil
    shutil.copyfile(out/'05_combined_DEP_all_raw.png',out/'experimental_DEP_fit.png')
    print('[NUTS] diagnostic pass:',diag['nuts_pass'],'GA-centered joint 4/4 posterior mass:',round(joint,4))
    print('[OUTPUT]',out)
    print('[WARNING] No empirical truth-recovery rates / posterior coverage can be computed on the experiment.')


if __name__=='__main__':main()
