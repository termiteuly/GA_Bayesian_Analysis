#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Posterior and predictive reporting for dep_experiment_v5_integrated.py.

No change to the V5 single-shell forward model, priors, likelihood, GA, FIM,
NUTS diagnostics or GA/FIM/Bayes acceptance gates is made by this module.
All spectral plots show every raw, finite individual observation.
Experimental coverage here is in-sample posterior predictive checking, NOT
coverage of unknown dielectric truth and NOT an out-of-sample calibration.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import ndtr
from matplotlib.ticker import MaxNLocator, FuncFormatter
from dep_synthetic_full_v5 import EPS0, PARAM_NAMES, LN_LOW, LN_HIGH, FREQ_6, ga_velocity_numpy

Q=(.025, .25, .50, .75, .975)
PARAM_DISPLAY=(r'log10(eps_cyto / eps0)',r'log10(sigma_cyto / S m-1)',
               r'log10(eps_shell / eps0)',r'log10(sigma_shell / S m-1)')


def log_posterior_at_draws(m, sigma, raw, freq, idx, noise_model, batch_size=256):
    """Joint log density for theta=log(dielectric params), sigma in original scale.

    Four *log* dielectric parameters have constant Uniform prior density in
    their allowed domain. Sigma has HalfCauchy(scale=5) density in original
    positive-sigma coordinates; this matches the NumPyro model's sample sites.
    This is a sample-based approximate MAP, NOT global optimization.
    """
    m=np.asarray(m,float); sig=np.asarray(sigma,float)
    if sig.ndim==1: sig=sig[:,None]
    y=np.concatenate([np.asarray(v,float) for v in raw])
    inds=np.concatenate([np.full(len(v),k,dtype=int) for k,v in enumerate(raw)])
    score=np.full(len(m),-np.inf)
    const=-np.log(LN_HIGH-LN_LOW).sum()
    for a in range(0,len(m),batch_size):
        b=min(len(m),a+batch_size); th=m[a:b]; ss=sig[a:b]
        good=(np.all((th>=LN_LOW)&(th<=LN_HIGH),axis=1) &
              np.all(np.isfinite(ss)&(ss>0),axis=1))
        if not np.any(good): continue
        pred=ga_velocity_numpy(freq,th)
        if noise_model=='common':
            scale=ss[:,0,None]
            prior=np.log(2/(np.pi*5))-np.log1p((ss[:,0]/5)**2)
        else:
            scale=ss[:,inds]
            prior=np.sum(np.log(2/(np.pi*5))-np.log1p((ss/5)**2),axis=1)
        mu=pred[:,inds]
        s=scale if noise_model=='per_frequency' else np.broadcast_to(scale,mu.shape)
        ll=-0.5*np.sum(((y[None,:]-mu)/s)**2+2*np.log(s)+np.log(2*np.pi),axis=1)
        v=ll+prior+const
        score[a:b]=np.where(good,v,-np.inf)
    return score


def choose_approx_map(m,sigma,raw,noise_model,seed,max_candidates=5000):
    rng=np.random.default_rng(seed+151)
    indices=rng.choice(len(m),size=min(max_candidates,len(m)),replace=False)
    vals=log_posterior_at_draws(m[indices],sigma[indices],raw,FREQ_6,None,noise_model)
    if not np.any(np.isfinite(vals)):
        raise RuntimeError('No finite log joint density among MAP candidates')
    loc=int(np.nanargmax(vals));return int(indices[loc]),float(vals[loc]),len(indices)


def _predictive_sample(lat, sig, counts, nnoise, seed):
    """Monte Carlo predictive draws at observed frequencies (S*M,K)."""
    lat=np.asarray(lat,float);s=np.asarray(sig,float)
    S,K=lat.shape
    if s.ndim==1:s=s[:,None]
    if s.shape[1]==1:s=np.broadcast_to(s,(S,K))
    if s.shape!=(S,K):raise ValueError(f'Noise array shape {s.shape}, expected ({S},{K})')
    rng=np.random.default_rng(seed)
    loc=np.repeat(lat,nnoise,axis=0)
    scales=np.repeat(s,nnoise,axis=0)
    indiv=loc+scales*rng.standard_normal(loc.shape)
    mean=loc+(scales/np.sqrt(np.asarray(counts)[None,:]))*rng.standard_normal(loc.shape)
    return indiv,mean


def _interval(arr):
    return np.quantile(arr,Q,axis=0)


def gaussian_mixture_quantiles(means, sigmas, reference_n=1, iterations=36):
    """Smooth, deterministic quantiles of equal-weight Normal-mixture PPC.

    For every frequency, F(y) = mean_s Phi((y-mu_s)/(sigma_s/sqrt(n))).
    This is Rao-Blackwellized conditional on retained parameter/noise draws,
    and avoids Monte Carlo noise flicker in dense-frequency credible bands.
    """
    mu=np.asarray(means,float);s=np.asarray(sigmas,float)
    if mu.ndim!=2:raise ValueError('means must be (posterior_draws, frequencies)')
    if s.ndim==1:s=s[:,None]
    if s.shape[1]==1:s=np.broadcast_to(s,mu.shape)
    if s.shape!=mu.shape:raise ValueError('noise scale and mean shape mismatch')
    if np.any(s<=0) or not np.all(np.isfinite(s)):
        raise ValueError('Posterior noise scale must be finite and positive')
    n=np.asarray(reference_n,float)
    s=s/np.sqrt(n)
    lo=np.broadcast_to(np.min(mu-9*s,axis=0),(len(Q),mu.shape[1])).copy()
    hi=np.broadcast_to(np.max(mu+9*s,axis=0),(len(Q),mu.shape[1])).copy()
    target=np.asarray(Q)[:,None]
    for _ in range(iterations):
        middle=(lo+hi)/2
        cdf=np.mean(ndtr((middle[None,:,:]-mu[:,None,:])/s[:,None,:]),axis=0)
        too_low=cdf<target
        lo=np.where(too_low,middle,lo)
        hi=np.where(too_low,hi,middle)
    return (lo+hi)/2


def _metric(obs,pred):
    o=np.asarray(obs,float);p=np.asarray(pred,float)
    residual=o-p
    rmse=float(np.sqrt(np.mean(residual**2)))
    sst=float(np.sum((o-np.mean(o))**2))
    r2=float(1-np.sum(residual**2)/sst) if sst>0 else float('nan')
    return rmse,r2


def _logfreq_band_names(freq):
    f=np.log10(np.asarray(freq,float))
    q1,q2=np.quantile(f,[1/3,2/3])
    return np.where(f<=q1,'low',np.where(f<=q2,'mid','high'))


def _coverage(raw,means,counts,quant_lat,quant_indiv,quant_mean,out):
    # Quantiles order: 2.5,25,50,75,97.5. These are all IN-SAMPLE checks.
    freqrows=[]
    for k,values in enumerate(raw):
        n=int(len(values))
        record={'frequency_hz':float(FREQ_6[k]),'n_raw':n,'observed_mean':float(means[k])}
        for name,q,reference in [
            ('individual',quant_indiv,values),
            ('measurement_mean',quant_mean,np.array([means[k]])),
            ('latent_vs_observed_mean',quant_lat,np.array([means[k]]))]:
            for label,lo,hi in [('50',q[1,k],q[3,k]),('95',q[0,k],q[4,k])]:
                rate=np.mean((reference>=lo)&(reference<=hi))
                record[f'{name}_coverage{label}']=float(rate)
                record[f'{name}_hits{label}']=int(np.sum((reference>=lo)&(reference<=hi)))
        freqrows.append(record)
    freqdf=pd.DataFrame(freqrows)
    freqdf['logfrequency_band']=_logfreq_band_names(FREQ_6)
    freqdf.to_csv(out/'coverage_by_frequency.csv',index=False)
    def summarize(df,band):
        rows=[]
        total_raw=int(df.n_raw.sum());nfreq=len(df)
        for name in ('individual','measurement_mean','latent_vs_observed_mean'):
            for level in ('50','95'):
                hits=int(df[f'{name}_hits{level}'].sum())
                denom=total_raw if name=='individual' else nfreq
                rows.append(dict(band=band,level=name,interval_percent=int(level),
                    n_unit=denom,n_raw=total_raw,n_frequencies=nfreq,
                    hits=hits,coverage=hits/denom,
                    evaluation='IN_SAMPLE: not empirical coverage of parameter truth'))
        return rows
    aggregate=summarize(freqdf,'all')
    bandrows=[]
    for band in ('low','mid','high'):
        sub=freqdf[freqdf.logfrequency_band==band]
        if len(sub):bandrows+=summarize(sub,band)
    pd.DataFrame(aggregate).to_csv(out/'coverage_summary.csv',index=False)
    pd.DataFrame(bandrows).to_csv(out/'coverage_by_frequency_band.csv',index=False)
    return freqdf,pd.DataFrame(aggregate),pd.DataFrame(bandrows)


def _raw_scatter(ax,raw,jitter_log10,seed=42):
    # Plot EVERY point; random jitter is DISPLAY ONLY and NOT applied to data,
    # likelihood, fitting, FIM, frequency_summary or coverage.
    rng=np.random.default_rng(seed+112)
    for i,v in enumerate(raw):
        logx=np.log10(FREQ_6[i]/1e6)
        x=np.power(10.,logx+rng.uniform(-jitter_log10,jitter_log10,size=len(v)))
        ax.scatter(x,v,color='#51788F',s=11,alpha=.27,linewidths=0,
                   label='All raw velocity observations' if i==0 else '_nolegend_',zorder=2)


def _axes(ax, title, jitter):
    ax.set_xscale('log');ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('DEP velocity (um/s)');ax.set_title(title)
    ax.set_xlim(FREQ_6[0]/1e6*.77,FREQ_6[-1]/1e6*1.28)
    ax.grid(True,which='both',ls=':',alpha=.25)
    if jitter>0:
        ax.text(.01,.01,f'Plot-only horizontal jitter +/-{jitter:g} log10 decades',
                transform=ax.transAxes,fontsize=7,color='#666666')


def _obs_mean(ax,means,sd,counts,show_se=True):
    se=np.asarray(sd,float)/np.sqrt(counts)
    valid=np.isfinite(se)
    if show_se and np.any(valid):
        ax.errorbar(FREQ_6[valid]/1e6,np.asarray(means)[valid],yerr=se[valid],
                    fmt='o',markersize=4,capsize=3,color='#111111',zorder=6,
                    label='Observed mean +/- SE')
    if np.any(~valid):
        ax.plot(FREQ_6[~valid]/1e6,np.asarray(means)[~valid],'o',color='black',
                markersize=4,zorder=6,label='Observed mean (n=1)')


def _save(fig,out,name):
    fig.tight_layout();fig.savefig(out/name,dpi=190,bbox_inches='tight');plt.close(fig)


def _plot_spectra(out,a,raw,means,sd,counts,dense,curves,obs_q,dense_q,common,diag):
    jd=a.plot_jitter_log10
    def new(title):
        fig,ax=plt.subplots(figsize=(10,6))
        _raw_scatter(ax,raw,jd,a.seed)
        _obs_mean(ax,means,sd,counts)
        _axes(ax,title,jd)
        if not diag['nuts_pass']:
            ax.text(.98,.98,'NUTS DIAGNOSTICS FAILED - EXPLORATORY ONLY',
                    transform=ax.transAxes,va='top',ha='right',color='#B91C1C',
                    fontsize=9,fontweight='bold',zorder=10)
        return fig,ax
    # 1: all point estimates, clearly distinct definitions.
    fig,ax=new('DEP spectra: all observations and alternative fitting curves')
    colors={'GA':'#D97706','posterior_median_parameters':'#713E99',
            'posterior_logmean_parameters':'#64748B',
            'posterior_mean_response':'#0F766E','approx_sample_MAP':'#BE185D'}
    for key,label,ls in [('GA','Multi-start GA','-'),
                         ('posterior_median_parameters','Posterior median parameters','-.'),
                         ('posterior_logmean_parameters','Geometric mean parameters',':'),
                         ('posterior_mean_response','Posterior mean response','-'),
                         ('approx_sample_MAP','Sample-based approx. MAP','--')]:
        ax.plot(dense/1e6,curves[key],ls,lw=2,color=colors[key],label=label,zorder=4)
    ax.legend(fontsize=8);_save(fig,out,'01_fitting_curves_all_raw.png')
    # 2: latent interval (parameter uncertainty ONLY)
    fig,ax=new('Latent DEP response: posterior credible interval (not prediction interval)')
    ax.fill_between(dense/1e6,dense_q['latent'][0],dense_q['latent'][4],
                    color='#0F766E',alpha=.21,label='95% latent credible band')
    ax.fill_between(dense/1e6,dense_q['latent'][1],dense_q['latent'][3],
                    color='#0F766E',alpha=.22,label='50% latent credible band')
    ax.plot(dense/1e6,curves['posterior_mean_response'],color='#0F766E',lw=2,label='Posterior mean response')
    ax.legend(fontsize=8);_save(fig,out,'02_latent_CI_all_raw.png')
    # 3: individual predictive interval, evaluated with residual noise.
    fig,ax=new('Individual-observation posterior predictive intervals (in-sample)')
    if common:
        ax.fill_between(dense/1e6,dense_q['individual'][0],dense_q['individual'][4],
                        color='#2563EB',alpha=.18,label='95% individual PI')
        ax.fill_between(dense/1e6,dense_q['individual'][1],dense_q['individual'][3],
                        color='#2563EB',alpha=.16,label='50% individual PI')
    else:
        ax.vlines(FREQ_6/1e6,obs_q['individual'][0],obs_q['individual'][4],
                  color='#2563EB',linewidth=7,alpha=.22,label='95% PI at measured frequencies only')
        ax.vlines(FREQ_6/1e6,obs_q['individual'][1],obs_q['individual'][3],
                  color='#2563EB',linewidth=4,alpha=.36,label='50% PI at measured frequencies only')
    ax.plot(dense/1e6,curves['posterior_mean_response'],color='#0F766E',lw=2,label='Posterior mean response')
    ax.legend(fontsize=8);_save(fig,out,'03_individual_PPC_all_raw.png')
    # 4: mean-level PI uses true n_k at observed frequencies. The dense band
    # for common noise is illustrative only, with a stated reference n.
    fig,ax=new('Replicate-mean posterior predictive intervals (in-sample)')
    if common:
        ax.fill_between(dense/1e6,dense_q['measurement_mean'][0],dense_q['measurement_mean'][4],
                        color='#7C3AED',alpha=.12,
                        label=f'95% mean PI (dense, reference n={int(np.median(counts))})')
    ax.vlines(FREQ_6/1e6,obs_q['measurement_mean'][0],obs_q['measurement_mean'][4],
              color='#7C3AED',alpha=.65,linewidth=4,label='95% mean PI (actual n_k)')
    ax.vlines(FREQ_6/1e6,obs_q['measurement_mean'][1],obs_q['measurement_mean'][3],
              color='#7C3AED',alpha=.8,linewidth=2,label='50% mean PI (actual n_k)')
    ax.plot(dense/1e6,curves['posterior_mean_response'],color='#0F766E',lw=2,label='Posterior mean response')
    ax.legend(fontsize=8);_save(fig,out,'04_mean_PPC_all_raw.png')
    # 5: one synthesis plot requested for publication-style overview.
    fig,ax=new('GA + Bayesian DEP inference: mean response and predictive uncertainty')
    if common:
        ax.fill_between(dense/1e6,dense_q['individual'][0],dense_q['individual'][4],
                        color='#2563EB',alpha=.11,label='95% individual PI')
    else:
        ax.vlines(FREQ_6/1e6,obs_q['individual'][0],obs_q['individual'][4],
                  color='#2563EB',linewidth=6,alpha=.18,label='95% individual PI (observed only)')
    ax.fill_between(dense/1e6,dense_q['latent'][0],dense_q['latent'][4],
                    color='#0F766E',alpha=.20,label='95% latent credible band')
    ax.plot(dense/1e6,curves['GA'],color='#D97706',lw=2,label='GA')
    ax.plot(dense/1e6,curves['posterior_mean_response'],color='#0F766E',lw=2,label='Posterior mean response')
    ax.legend(fontsize=8);_save(fig,out,'05_combined_DEP_all_raw.png')


def _contour_kde(ax,x,y,colors=('#0D9488','#155E75')):
    """Approximate 50/95% *bivariate* HDR contours, NOT 4D credible sets."""
    if len(x)<25 or np.std(x)<1e-8 or np.std(y)<1e-8: return
    try:
        xy=np.vstack((x,y));kde=gaussian_kde(xy)
        pad_x=max(float(np.ptp(x))*.10,.03);pad_y=max(float(np.ptp(y))*.10,.03)
        xx,yy=np.mgrid[(x.min()-pad_x):(x.max()+pad_x):55j,
                       (y.min()-pad_y):(y.max()+pad_y):55j]
        z=kde(np.vstack((xx.ravel(),yy.ravel()))).reshape(xx.shape)
        flat=np.sort(z.ravel())[::-1]
        mass=np.cumsum(flat)/np.sum(flat)
        thresholds=[float(flat[min(np.searchsorted(mass,t),len(flat)-1)]) for t in (.95,.50)]
        if thresholds[0]<thresholds[1]:
            ax.contour(xx,yy,z,levels=thresholds,colors=list(colors),linewidths=[1.0,1.6])
    except (np.linalg.LinAlgError,ValueError, FloatingPointError):
        pass


def _plotting_options(a):
    """Visualization-only options; never change posterior samples or decisions."""
    style=getattr(a,'marginal_style','kde')
    bw=float(getattr(a,'marginal_kde_bandwidth',1.0))
    max_draws=int(getattr(a,'marginal_max_draws',8000))
    if style not in ('kde','hist+kde'):
        raise ValueError('marginal_style must be kde or hist+kde')
    if not np.isfinite(bw) or not (0.2 <= bw <= 5):
        raise ValueError('marginal_kde_bandwidth must be within [0.2,5]')
    if max_draws < 100:
        raise ValueError('marginal_max_draws must be >=100')
    return style,bw,max_draws


def _density_curve(ax, values, color, *, bw=1.0, style='kde',
                   draw_limit=8000, seed=0, label=None):
    """Smooth sample KDE in PLOTTED x coordinates (normalized to density).

    No draws are altered. For performance, KDE visualization alone can use a
    reproducible sample of posterior draws; the scientific summaries use all.
    """
    arr=np.asarray(values,float).reshape(-1)
    arr=arr[np.isfinite(arr)]
    if arr.size < 2:
        ax.text(.5,.5,'Insufficient samples',transform=ax.transAxes,
                ha='center',va='center',fontsize=9)
        return
    if arr.size > draw_limit:
        rng=np.random.default_rng(seed)
        arr=arr[rng.choice(arr.size,draw_limit,replace=False)]
    low,high=np.percentile(arr,[.1,99.9])
    span=max(high-low,1e-12)
    xlow=low-.05*span
    xhigh=high+.05*span
    # Avoid an awkward all-zero density when samples are nearly constant.
    if np.ptp(arr) < 1e-10*max(1.0,float(abs(np.median(arr)))):
        ax.axvline(float(np.median(arr)),color=color,lw=2,label=label)
        ax.set_xlim(float(np.median(arr))-.5,float(np.median(arr))+.5)
        return
    if style=='hist+kde':
        ax.hist(arr,bins=38,density=True,color=color,alpha=.11,
                edgecolor='none',label='_nolegend_')
    try:
        kde=gaussian_kde(arr,bw_method=lambda k: k.scotts_factor()*bw)
        grid=np.linspace(xlow,xhigh,350)
        pdf=kde(grid)
        if np.all(np.isfinite(pdf)) and np.any(pdf>0):
            ax.fill_between(grid,0,pdf,color=color,alpha=.20,zorder=2)
            ax.plot(grid,pdf,color=color,lw=2.0,label=label,zorder=3)
        else:
            raise ValueError('Nonfinite KDE density')
    except (np.linalg.LinAlgError,ValueError,FloatingPointError):
        # Retain visible output even when KDE covariance is degenerate.
        ax.hist(arr,bins=30,density=True,histtype='step',color=color,
                linewidth=1.7,label=label)
    ax.set_xlim(xlow,xhigh)


def _compact_axis_data(values, kind, idx=None):
    """Produce short, honest x ticks without repeating long scientific notation.

    For narrow positive distributions use linear ticks after multiplying by a
    power of ten stated in the axis label. For broad ones plot log10(values)
    with an explicit log10 axis label. KDE density is wrt this plotted x.
    """
    vals=np.asarray(values,float).reshape(-1)
    if not np.all(np.isfinite(vals)) or np.any(vals<=0):
        raise ValueError('Posterior physical parameters must be finite, positive')
    if kind=='epsilon':
        rel=vals/EPS0
        lo,hi=np.percentile(rel,[.5,99.5])
        name=r'\epsilon_{\mathrm{cyto}}' if idx==0 else r'\epsilon_{\mathrm{shell}}'
        if hi/max(lo,1e-300)>100:
            return np.log10(rel),rf'$\log_{{10}}({name}/\epsilon_0)$','log10',0
        return rel,rf'${name}/\epsilon_0$','linear',0
    if kind=='noise':
        name=r'\sigma_y'
        unit=r'\mu\mathrm{m/s}'
    else:
        name=r'\sigma_{\mathrm{cyto}}' if idx==1 else r'\sigma_{\mathrm{shell}}'
        unit=r'\mathrm{S/m}'
    lo,hi=np.percentile(vals,[.5,99.5])
    if hi/max(lo,1e-300)>100:
        return np.log10(vals),rf'$\log_{{10}}({name}/({unit}))$','log10',0
    median=float(np.median(vals))
    exponent=int(np.floor(np.log10(median)))
    # Keep approximately human-size quantities unscaled, e.g. sigma_y=13.
    if 0.1<=median<100:
        exponent=0 if kind=='noise' else exponent
    exponent=max(-300,min(300,exponent))
    factor=10.0**exponent
    if exponent:
        label=rf'${name}\;(10^{{{exponent}}}\,{unit})$'
    else:
        label=rf'${name}\;({unit})$'
    return vals/factor,label,'linear',exponent


def _concise_ticks(ax, nbins=4):
    """Fixed-width decimal tick text, capped at 4-5 ticks, no axis offset."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins,min_n_ticks=2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f'{x:.3g}'))
    ax.tick_params(axis='x',labelsize=9,pad=4)
    ax.tick_params(axis='y',labelsize=9)
    ax.get_xaxis().get_offset_text().set_visible(False)


def _plot_corner(out,a,m,ga_theta,diag):
    style,bw,max_draws=_plotting_options(a)
    rng=np.random.default_rng(a.seed+505)
    sel=rng.choice(len(m),size=min(a.corner_max_draws,len(m)),replace=False)
    x=m[sel]/math.log(10)
    x[:,0]-=math.log10(EPS0);x[:,2]-=math.log10(EPS0)
    g=np.log10(ga_theta).copy()
    g[[0,2]]-=math.log10(EPS0)
    corr=np.corrcoef(m,rowvar=False)
    fig,axs=plt.subplots(4,4,figsize=(12.8,12.2))
    for i in range(4):
        for j in range(4):
            ax=axs[i,j]
            if i==j:
                _density_curve(ax,x[:,i],'#2C7A7B',bw=bw,style=style,
                    draw_limit=max_draws,seed=a.seed+800+i)
                ax.axvline(g[i],color='#D97706',lw=1.7,ls='--')
                ax.axvline(float(np.median(x[:,i])),color='black',lw=1,ls=':')
                ax.set_yticks([])
            elif i>j:
                ax.scatter(x[:,j],x[:,i],s=2.5,alpha=.055,color='#2C7A7B',rasterized=True)
                _contour_kde(ax,x[:,j],x[:,i])
                ax.plot(g[j],g[i],marker='x',ms=8,mew=1.8,color='#D97706')
            else:
                ax.text(.5,.55,f'r = {corr[i,j]:+.2f}',ha='center',va='center',
                        transform=ax.transAxes,fontsize=12,color='#334155')
                ax.axis('off')
            if i>=j:
                _concise_ticks(ax,nbins=3)
            if i==3:ax.set_xlabel(PARAM_DISPLAY[j],fontsize=9)
            else:ax.tick_params(axis='x',labelbottom=False)
            if j==0 and i>0:ax.set_ylabel(PARAM_DISPLAY[i],fontsize=9)
            elif j>0:ax.tick_params(axis='y',labelleft=False)
            ax.tick_params(axis='both',labelsize=8)
    fig.suptitle('Experimental joint posterior (4 dielectric parameters)  |  GA: orange x  |  '
                 'median: black dotted (diagonal)  |  50/95% bivariate contours',fontsize=12,y=.995)
    if not diag['nuts_pass']:
        fig.text(.5,.004,'WARNING: NUTS DIAGNOSTICS FAILED; exploratory samples only, not validated posterior',
                 ha='center',color='darkred',fontsize=11)
    _save(fig,out,'06_joint_posterior_corner.png')


def _plot_marginal(out,a,m,sigma,noise_model):
    """KDE posterior panels with compact axis labels and no overlapping ticks."""
    style,bw,max_draws=_plotting_options(a)
    fig,ax=plt.subplots(3,2,figsize=(13.5,11.2))
    axes=ax.ravel()
    for i in range(4):
        vals=np.exp(m[:,i])
        kind='epsilon' if i in (0,2) else 'conductivity'
        coord,label,scale_kind,exponent=_compact_axis_data(vals,kind,i)
        _density_curve(axes[i],coord,'#2C7A7B',bw=bw,style=style,
                       draw_limit=max_draws,seed=a.seed+1200+i)
        axes[i].set_xlabel(label,fontsize=11,labelpad=8)
        axes[i].set_ylabel('Density',fontsize=10)
        axes[i].set_title('Posterior: '+PARAM_NAMES[i],fontsize=11,pad=9)
        axes[i].grid(alpha=.13,axis='y')
        _concise_ticks(axes[i])
    s=np.asarray(sigma,float)
    if noise_model=='common':
        coord,label,_,_=_compact_axis_data(s.reshape(-1),'noise')
        _density_curve(axes[4],coord,'#5A7B9F',bw=bw,style=style,
                       draw_limit=max_draws,seed=a.seed+1400)
        axes[4].set_title('Posterior: sigma_y',fontsize=11,pad=9)
        axes[4].set_xlabel(label,fontsize=11,labelpad=8)
    else:
        # All six frequencies in ONE panel using the same plotting coordinate.
        pooled=s.reshape(-1)
        _,xlabel,kind,exp_=_compact_axis_data(pooled,'noise')
        union_low,union_high=float('inf'),-float('inf')
        for i in range(s.shape[1]):
            if kind=='log10':coord=np.log10(s[:,i])
            else: coord=s[:,i]/(10.0**exp_)
            _density_curve(axes[4],coord,f'C{i%10}',bw=bw,style=style,
                           draw_limit=max_draws,seed=a.seed+1400+i,
                           label=f'{FREQ_6[i]/1e6:g} MHz')
            lo_,hi_=axes[4].get_xlim()
            union_low=min(union_low,lo_)
            union_high=max(union_high,hi_)
        axes[4].set_xlim(union_low,union_high)
        axes[4].legend(fontsize=8,ncol=2,frameon=False)
        axes[4].set_title('Posterior: sigma_by_frequency',fontsize=11,pad=9)
        axes[4].set_xlabel(xlabel,fontsize=11,labelpad=8)
    axes[4].set_ylabel('Density',fontsize=10)
    axes[4].grid(alpha=.13,axis='y')
    _concise_ticks(axes[4])
    axes[5].axis('off')
    axes[5].text(0,.9,'Smooth Gaussian KDE (Scott bandwidth x '+f'{bw:g}'+')',fontsize=10,
                 color='#475569',transform=axes[5].transAxes)
    axes[5].text(0,.75,'Posterior draws are unchanged.\n'
                 'X-axis scaling is indicated in each panel.\n'
                 'Y-axis: probability density (not histogram counts).',
                 fontsize=9,color='#475569',va='top',transform=axes[5].transAxes)
    fig.suptitle('Experimental posterior marginals — smooth density',fontsize=15,y=.996)
    fig.subplots_adjust(hspace=.4,wspace=.25)
    _save(fig,out,'07_posterior_marginals.png')

def _plot_trace(out,mcmc,diag):
    try:
        import arviz as az
        names=['log_'+p for p in PARAM_NAMES]
        names+=['sigma_y' if diag['noise_model']=='common' else 'sigma_by_frequency']
        axes=az.plot_trace(az.from_numpyro(mcmc),var_names=names,compact=True)
        fig=np.asarray(axes).ravel()[0].figure
        fig.suptitle('NUTS chain traces and marginal distributions',y=1.002)
        _save(fig,out,'08_nuts_trace.png')
    except Exception as e:
        (out/'TRACE_PLOT_WARNING.txt').write_text(f'Could not generate trace: {e}\n',encoding='utf8')


def _plot_residuals(out,a,raw,means,curves_observed):
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    pred=curves_observed['posterior_mean_response']
    for i,rs in enumerate(raw):
        y=np.asarray(rs)-pred[i]
        axs[0].scatter(np.full(len(rs),FREQ_6[i]/1e6),y,s=12,alpha=.27,color='#51788F')
    axs[0].axhline(0,color='black',lw=1)
    axs[0].set_xscale('log');axs[0].set_xlabel('Frequency (MHz)')
    axs[0].set_ylabel('Observed - posterior mean (um/s)');axs[0].set_title('All raw residuals')
    axs[1].plot(FREQ_6/1e6,np.asarray(means)-pred,'o-',color='#0F766E')
    axs[1].axhline(0,color='black',lw=1)
    axs[1].set_xscale('log');axs[1].set_xlabel('Frequency (MHz)')
    axs[1].set_ylabel('Mean residual (um/s)');axs[1].set_title('Frequency-mean residuals')
    _save(fig,out,'09_residual_diagnostics.png')


def extended_sampler_diagnostics(out,a,mcmc):
    """Additional diagnostics. Does NOT redefine the original V5 NUTS gate."""
    if mcmc is None:return
    import arviz as az
    idata=az.from_numpyro(mcmc)
    summ=az.summary(idata)
    extras=mcmc.get_extra_fields()
    answer={'note':'Supplementary sampler diagnostics; original V5 gate unchanged',
            'init_strategy':a.init_strategy,'init_radius':a.init_radius if a.init_strategy=='uniform' else 0.0,
            'dense_mass':a.dense_mass,
            'max_rhat':float(summ['r_hat'].max()),
            'min_bulk_ess':float(summ['ess_bulk'].min()),
            'min_tail_ess':float(summ['ess_tail'].min())}
    ratio=(summ['mcse_mean']/summ['sd']).replace([np.inf,-np.inf],np.nan)
    answer['max_mcse_mean_over_sd']=float(ratio.max())
    total=a.chains*a.samples
    answer['min_bulk_ess_fraction']=float(summ['ess_bulk'].min()/total)
    answer['min_tail_ess_fraction']=float(summ['ess_tail'].min()/total)
    if 'num_steps' in extras:
        steps=np.asarray(extras['num_steps'],float).reshape(-1)
        upper=2**a.max_tree_depth-1
        answer.update(max_num_steps=int(np.max(steps)),p95_num_steps=float(np.percentile(steps,95)),
                      fraction_num_steps_ge_theoretical_max=float(np.mean(steps>=upper)),
                      fraction_num_steps_ge_90_percent_max=float(np.mean(steps>=0.9*upper)))
    if 'accept_prob' in extras:
        z=np.asarray(extras['accept_prob'],float)
        answer['accept_prob_median']=float(np.median(z))
    if 'energy' in extras:
        try:
            bf=np.asarray(az.bfmi(idata),float)
            answer['bfmi_by_chain']=bf.tolist()
            answer['bfmi_min']=float(np.min(bf))
        except Exception as e:answer['bfmi_unavailable']=str(e)
    (out/'NUTS_diagnostics_extended.json').write_text(json.dumps(answer,indent=2),encoding='utf8')


def make_extended_outputs(a,out,raw,counts,means,sd,ga_theta,samples,diag,mcmc):
    """Add temp_alldata analyses while retaining the original V5 scientific core."""
    out=Path(out)
    m=np.column_stack([np.asarray(samples['log_'+p],float).reshape(-1) for p in PARAM_NAMES])
    sigma=np.asarray(samples['sigma_y'] if a.noise_model=='common' else samples['sigma_by_frequency'],float)
    if sigma.ndim==1:sigma=sigma[:,None]
    if len(m)!=len(sigma):raise ValueError('Unaligned posterior draws')
    if a.ppc_draws<1 or a.ppc_noise_draws<1 or a.dense_points<10:
        raise ValueError('ppc-draws/noise draws must be positive and dense-points >=10')
    _plotting_options(a)
    if a.plot_jitter_log10<0 or a.plot_jitter_log10>.08:
        raise ValueError('plot-jitter-log10 must be within [0,0.08]')
    rng=np.random.default_rng(a.seed+70)
    pick=rng.choice(len(m),size=min(a.ppc_draws,len(m)),replace=False)
    vobs=ga_velocity_numpy(FREQ_6,m[pick])
    sind=sigma[pick]
    q_lat=_interval(vobs)
    if a.ppc_quantile_method=='analytic':
        q_ind=gaussian_mixture_quantiles(vobs,sind,reference_n=1)
        q_mean=gaussian_mixture_quantiles(vobs,sind,reference_n=counts)
    else:
        yraw,ymean=_predictive_sample(vobs,sind,counts,a.ppc_noise_draws,a.seed+301)
        q_ind=_interval(yraw);q_mean=_interval(ymean)
    if a.map_candidates<1:raise ValueError('--map-candidates must be >=1')
    midx,mlog,ncand=choose_approx_map(m,sigma,raw,a.noise_model,a.seed,a.map_candidates)
    theta_map=np.exp(m[midx]);theta_med=np.exp(np.median(m,axis=0))
    theta_logmean=np.exp(np.mean(m,axis=0))
    freq_dense=np.geomspace(FREQ_6[0],FREQ_6[-1],a.dense_points)
    vdense=ga_velocity_numpy(freq_dense,m[pick])
    q_lat_dense=_interval(vdense)
    dense_curves={
       'GA':ga_velocity_numpy(freq_dense,np.log(ga_theta)),
       'posterior_median_parameters':ga_velocity_numpy(freq_dense,np.log(theta_med)),
       'posterior_logmean_parameters':ga_velocity_numpy(freq_dense,np.log(theta_logmean)),
       'posterior_mean_response':np.mean(vdense,axis=0),
       'approx_sample_MAP':ga_velocity_numpy(freq_dense,np.log(theta_map))}
    observed_curves={
       'GA':ga_velocity_numpy(FREQ_6,np.log(ga_theta)),
       'posterior_median_parameters':ga_velocity_numpy(FREQ_6,np.log(theta_med)),
       'posterior_logmean_parameters':ga_velocity_numpy(FREQ_6,np.log(theta_logmean)),
       'posterior_mean_response':np.mean(vobs,axis=0),
       'approx_sample_MAP':ga_velocity_numpy(FREQ_6,np.log(theta_map))}
    dense_q={'latent':q_lat_dense}
    noise_median=np.median(sind,axis=0)
    common=(a.noise_model=='common')
    if common:
        # A dense frequency-specific observation-noise distribution is well
        # defined only for the global sigma model. For per-frequency noise,
        # prediction intervals are shown only at measured frequencies.
        nref=int(np.median(counts))
        if a.ppc_quantile_method=='analytic':
            dense_q['individual']=gaussian_mixture_quantiles(vdense,sind[:,0],1)
            dense_q['measurement_mean']=gaussian_mixture_quantiles(vdense,sind[:,0],nref)
        else:
            dnoise=np.random.default_rng(a.seed+302).standard_normal(
                (len(pick)*a.ppc_noise_draws,a.dense_points))
            dloc=np.repeat(vdense,a.ppc_noise_draws,axis=0)
            dsd=np.repeat(sind[:,0],a.ppc_noise_draws)[:,None]
            dense_q['individual']=_interval(dloc+dsd*dnoise)
            dnoise2=np.random.default_rng(a.seed+303).standard_normal(dnoise.shape)
            dense_q['measurement_mean']=_interval(dloc+(dsd/np.sqrt(nref))*dnoise2)
    else:
        nref=None
    for arrays,name,axis in [(q_lat,'latent',0),(q_ind,'individual',0),(q_mean,'mean',0)]:
        if arrays.shape!=(5,6):raise AssertionError(f'Bad {name} band shape: {arrays.shape}')
    obs_table=pd.DataFrame(dict(frequency_hz=FREQ_6,n_raw=counts,
        observed_mean=means,observed_sd=sd,
        observed_SE=np.asarray(sd)/np.sqrt(counts)))
    for key,v in observed_curves.items():obs_table[key]=v
    for name,q in [('latent',q_lat),('individual_predictive',q_ind),('replicate_mean_predictive',q_mean)]:
        for level,ix in [('q025',0),('q25',1),('q50',2),('q75',3),('q975',4)]:
            obs_table[f'{name}_{level}']=q[ix]
    obs_table.to_csv(out/'posterior_curves_observed_frequency.csv',index=False)
    dense_table=pd.DataFrame({'frequency_hz':freq_dense})
    for key,v in dense_curves.items():dense_table[key]=v
    for name,q in [('latent',q_lat_dense)]+([(k,v) for k,v in dense_q.items() if k!='latent']):
        for label,idx in [('q025',0),('q25',1),('q50',2),('q75',3),('q975',4)]:
            dense_table[f'{name}_{label}']=q[idx]
    dense_table.to_csv(out/'posterior_curves_dense_frequency.csv',index=False)
    rows=[];observed_raw=np.concatenate(raw)
    for key,pred in observed_curves.items():
        y_all=np.concatenate([np.full(len(v),pred[i]) for i,v in enumerate(raw)])
        for scope,obs,p in [('all_individual_measurements',observed_raw,y_all),
                            ('unweighted_frequency_means',means,pred)]:
            rmse,r2=_metric(obs,p)
            rows.append(dict(curve=key,comparison=scope,rmse_um_s=rmse,R_squared=r2,
                    n_points=len(obs),note='In-sample descriptive fit, not held-out validation'))
    pd.DataFrame(rows).to_csv(out/'fitting_metrics.csv',index=False)
    _,cov_summary,_=_coverage(raw,means,counts,q_lat,q_ind,q_mean,out)
    # Strictly separate approximate *sample-based* MAP from numerical MAP.
    (out/'approximate_MAP.json').write_text(json.dumps(dict(
        definition='Highest joint density among a random subset of posterior draws in log dielectric coordinates and original noise coordinates; NOT numerical/global MAP',
        candidate_draws=ncand,sample_index=midx,joint_log_density=mlog,
        parameter_names=list(PARAM_NAMES),dielectric_parameters=theta_map.tolist(),
        noise_model=a.noise_model,noise_value=sigma[midx].tolist()),indent=2),encoding='utf8')
    corr=np.corrcoef(m,rowvar=False)
    # V5 already writes posterior_correlation_ln.csv; repeat no alternative definition.
    _plot_spectra(out,a,raw,means,sd,counts,freq_dense,dense_curves,
                  dict(latent=q_lat,individual=q_ind,measurement_mean=q_mean),dense_q,common,diag)
    _plot_corner(out,a,m,ga_theta,diag)
    _plot_marginal(out,a,m,sigma,a.noise_model)
    _plot_trace(out,mcmc,diag)
    extended_sampler_diagnostics(out,a,mcmc)
    _plot_residuals(out,a,raw,means,observed_curves)
    (out/'EXTENDED_ANALYSIS_NOTES.json').write_text(json.dumps(dict(
        note='Original V5 GA/FIM/Bayesian reporting gates are unchanged',
        initialization=a.init_strategy,uniform_radius=a.init_radius,dense_mass=a.dense_mass,
        posterior_draws_total=len(m),posterior_draws_used=len(pick),
        predictive_noise_draws_per_posterior=a.ppc_noise_draws,
        predictive_quantile_method=a.ppc_quantile_method,
        dense_grid_points=a.dense_points,mean_dense_reference_n=nref,
        raw_points_plotted_in_every_spectral_figure=int(counts.sum()),
        horizontal_plot_jitter_log10=a.plot_jitter_log10,
        in_sample_PPC_not_out_of_sample=True,
        latent_coverage_vs_measured_mean_not_parameter_truth=True,
        per_frequency_noise_dense_PI='OMITTED; no interpolation of frequency-specific sigmas' if not common else 'defined by global sigma',
        joint_contours='Bivariate KDE 50%/95% highest density approximations, not 4-dimensional credible regions',
        marginal_style=getattr(a,'marginal_style','kde'),
        marginal_kde_bandwidth_factor=getattr(a,'marginal_kde_bandwidth',1.0),
        marginal_kde_draw_limit=getattr(a,'marginal_max_draws',8000),
        marginal_plot_yaxis='DENSITY (not histogram counts)',
        map_warning='Approximate highest-density posterior draw only, see approximate_MAP.json',
        convergence_pass=bool(diag['nuts_pass'])),indent=2),encoding='utf8')
    print('[EXTENDED] In-sample PPC/coverage, all-raw spectra, corner, MAP and fitting metrics saved')
    print('[EXTENDED] Individual 95% in-sample coverage:',float(cov_summary.loc[
        (cov_summary.level=='individual')&(cov_summary.interval_percent==95),'coverage'].iloc[0]))
