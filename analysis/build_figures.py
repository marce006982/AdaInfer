"""Regenerate the data-derived AdaInfer figures and table fragments."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
FIG=ROOT/'figures'; TAB=ROOT/'results'
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'axes.spines.top':False,
 'axes.spines.right':False,'savefig.bbox':'tight'})
colors={'q_learning':'#2878B5','q_online':'#2878B5','greedy_sla':'#E28732',
 'greedy_sla_stale':'#E28732','greedy_sla_ewma':'#CC6677','linucb':'#6A994E',
 'linucb_online':'#6A994E','adaptive_linucb':'#8D6CAB','dynamic_budget_q8':'#B49B37',
 'q_frozen':'#777777'}
names={'q_learning':'AdaInfer','q_online':'Online Q-learning','greedy_sla':'Greedy-SLA',
 'greedy_sla_stale':'Stale Greedy-SLA','greedy_sla_ewma':'EWMA Greedy-SLA',
 'linucb':'LinUCB','linucb_online':'Online LinUCB','adaptive_linucb':'Adaptive LinUCB',
 'dynamic_budget_q8':'Dynamic-budget Qwen3/Q8','q_frozen':'Frozen Q table',
 'oracle_reward':'Profile-greedy reward','oracle_adaptive':'Updated-profile reward',
 'hysteresis_rule':'Hysteresis rule','static_q4':'Static Qwen2.5/Q4','static_q8':'Static Qwen3/Q8',
 'greedy_latency':'Latency-greedy','greedy_quality':'Quality-greedy',
 'q_learning_empirical':'AdaInfer (empirical)'}
table_names=dict(names)
table_names.update({
 'dynamic_budget_q8':'Dynamic-budget Qwen3-0.6B/Q8\\_0',
 'static_q4':'Static Qwen2.5-0.5B/Q4\\_K\\_M',
 'static_q8':'Static Qwen3-0.6B/Q8\\_0'})
profile_names={
 'qwen2.5-0.5b-q4':'Qwen2.5-0.5B/Q4\\_K\\_M',
 'qwen3-0.6b-q8':'Qwen3-0.6B/Q8\\_0'}
def save(fig,name):
 fig.savefig(FIG/(name+'.pdf')); fig.savefig(FIG/(name+'.png'),dpi=180);plt.close(fig)
def pm(row,k,d=3,scale=1):
 return f"${row[k+'_mean']/scale:.{d}f}\\pm{row[k+'_std']/scale:.{d}f}$"
def write(name,rows): (TAB/name).write_text('\n'.join(r.rstrip(chr(92))+chr(92)*2 for r in rows)+'\n')

core=pd.read_csv(DATA/'core_metrics_workload_steady_16_32_64_128_20260513.csv')
core=core[core.device=='yahboom_nano']
fig,axs=plt.subplots(1,3,figsize=(7,2.35),layout='constrained')
for label,color,mark,group in [('Qwen2.5/Q4','#E28732','o','qwen2.5-0.5b-q4'),('Qwen3/Q8','#2878B5','^','qwen3-0.6b-q8')]:
 d=core[core.model_short==group].sort_values('token_budget')
 axs[0].errorbar(d.token_budget,d.latency_ms_mean/1000,yerr=d.latency_ms_std/1000,label=label,c=color,marker=mark,capsize=2)
 axs[1].plot(d.token_budget,d.ram_used_mb_mean/1024,c=color,marker=mark)
 axs[2].plot(d.token_budget,d.power_w_mean,c=color,marker=mark)
for ax,title,y in zip(axs,['(a) Inference latency','(b) Recorded RAM usage','(c) Board power'],['Seconds','GiB','Watts']):
 ax.set(title=title,xlabel='Generation budget',ylabel=y);ax.set_xticks([16,32,64,128]);ax.grid(alpha=.18)
axs[0].legend();save(fig,'profiles')
write('profiles.tex',[f"{profile_names[r.model_short]} & {int(r.token_budget)} & {r.latency_ms_mean/1000:.3f} & {r.latency_ms_std/1000:.3f} & {r.ram_used_mb_mean:.1f} & {r.power_w_mean:.3f} & {int(r.n)} \\\\" for _,r in core.iterrows()])

station=pd.read_csv(DATA/'rl_validation_aggregate_20260514.csv')
nom=station[station.scenario=='nominal'].set_index('strategy')
order=['q_learning','greedy_sla','linucb','oracle_reward','dynamic_budget_q8','hysteresis_rule','static_q4','static_q8']
write('stationary.tex',[table_names[p]+' & '+ ' & '.join([pm(nom.loc[p],'joint_sla_rate'),pm(nom.loc[p],'mean_total_latency_ms',3,1000),pm(nom.loc[p],'mean_infer_energy_j',2),pm(nom.loc[p],'mean_rl_reward')])+r' \\' for p in order])
req=pd.read_csv(DATA/'rl_validation_requests_20260514.csv');req=req[req.scenario=='nominal']
budget=req.groupby('strategy').selected_budget.value_counts(normalize=True).unstack(fill_value=0)
budget.to_csv(TAB/'stationary_budget_shares.csv')
write('budgets.tex',[table_names[p]+' & '+' & '.join(f'{100*budget.loc[p].get(b,0):.2f}' for b in [16,32,64,128])+r' \\' for p in ['q_learning','greedy_sla','linucb','dynamic_budget_q8']])
fig,axs=plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
for p in ['q_learning','greedy_sla','linucb','dynamic_budget_q8']:
 d=station[station.strategy==p].set_index('scenario').loc[['nominal','rtt_x1_5','burst_30','burst_40_loss','tight_deadline']]
 for ax,k,scale in [(axs[0],'joint_sla_rate',1),(axs[1],'mean_total_latency_ms',1000)]:
  ax.errorbar(np.arange(5),d[k+'_mean']/scale,yerr=d[k+'_std']/scale,c=colors[p],label=names[p],marker='o',capsize=2,lw=1)
for ax in axs:
 ax.set_xticks(np.arange(5),['Nominal','RTT ×1.5','Burst 30','Burst/loss','Tight'],rotation=20);ax.grid(alpha=.2)
axs[0].set_ylabel('Joint SLA');axs[1].set_ylabel('Latency (s)');axs[0].legend(fontsize=7,loc='lower left');save(fig,'stress')

emp=pd.read_csv(DATA/'empirical_quality_replay/empirical_quality_replay_summary_20260515.csv').set_index('strategy')
write('empirical.tex',[table_names[p]+' & '+' & '.join([pm(emp.loc[p],'joint_sla_rate'),pm(emp.loc[p],'quality_sla_rate'),pm(emp.loc[p],'mean_total_latency_ms',3,1000),pm(emp.loc[p],'mean_infer_energy_j',2)])+r' \\' for p in ['q_learning_empirical','greedy_sla','dynamic_budget_q8','static_q4','static_q8']])
ns=pd.read_csv(DATA/'nonstationary_corrected/rl_nonstationary_summary_20260514.csv')
rows=[]
for p in ['q_online','q_frozen','greedy_sla_stale','greedy_sla_ewma','linucb_online','adaptive_linucb','oracle_adaptive','dynamic_budget_q8']:
 pre=ns[(ns.strategy==p)&(ns.phase=='pre_shift')].iloc[0];post=ns[(ns.strategy==p)&(ns.phase=='post_shift')].iloc[0]
 rows.append(table_names[p]+' & '+f'{pre.joint_sla_rate_mean:.3f} & {pre.mean_latency_ms_mean/1000:.3f} & {pre.q8_action_rate_mean:.3f} & '+pm(post,'joint_sla_rate')+' & '+pm(post,'mean_latency_ms',3,1000)+' & '+f'{post.q8_action_rate_mean:.3f}'+r' \\')
write('drift.tex',rows)
trace=pd.read_csv(DATA/'nonstationary_corrected/rl_nonstationary_requests_20260514.csv')
fig,axs=plt.subplots(3,1,figsize=(7,5.8),sharex=True,layout='constrained')
for p in ['q_online','q_frozen','greedy_sla_ewma','adaptive_linucb']:
 d=trace[trace.strategy==p].copy();d['c8']=(d.model=='Qwen3-0.6B Q8').astype(float)
 d['block']=d.request_id//250
 g=d.groupby('block')[['joint_sla_met','total_latency_ms','c8']].mean()
 for ax,k,scale in [(axs[0],'joint_sla_met',1),(axs[1],'total_latency_ms',1000),(axs[2],'c8',1)]:
  ax.plot((g.index+1)*250,g[k]/scale,c=colors[p],label=names[p],lw=1.25)
for ax,lab in zip(axs,['Joint SLA','Latency (s)','Qwen3/Q8 selection fraction']):
 ax.axvline(5000,c='#555555',ls='--',lw=.8);ax.set_ylabel(lab);ax.grid(alpha=.2)
axs[0].set_ylim(.75,1.02);axs[2].set_ylim(.45,1.03);axs[0].legend(ncol=2,fontsize=8)
axs[-1].set_xlabel('Request index (non-overlapping 250-request blocks)');save(fig,'drift')
q=trace[trace.strategy=='q_online'];f=trace[trace.strategy=='q_frozen']
resets=trace[trace.strategy=='adaptive_linucb'].groupby(['seed','phase']).adaptive_resets.max()
resets.to_csv(TAB/'adaptive_reset_counts.csv')
ns.to_csv(TAB/'drift_full_summary.csv',index=False)

noise=pd.read_csv(DATA/'rl_quality_noise_summary_20260514.csv')
write('noise.tex',[f'{100*r.noise_rate:.0f} & '+pm(r,'joint_sla_rate')+' & '+pm(r,'mean_latency_ms',3,1000)+' & '+f'{r.q8_action_rate_mean:.3f}'+r' \\' for _,r in noise.iterrows()])
abl=pd.read_csv(DATA/'rl_reward_ablation_20260514.csv')
alabel=['SLA-focused','Without explicit quality gate','Without explicit switch penalty','Budget-retention variant']
write('ablation.tex',[alabel[i]+' & '+f'{r.joint_sla_rate:.3f} & {r.mean_total_latency_ms/1000:.3f} & {r.mean_infer_energy_j:.2f} & {r.mean_budget_retention:.3f}'+r' \\' for i,r in abl.iterrows()])
tim=pd.read_csv(DATA/'nonstationary_robustness/shift_timing_summary.csv')
tim.to_csv(TAB/'shift_timing_full.csv',index=False)

rtx=pd.read_csv(DATA/'rtx5060ti_local/profiling.csv')
rtx['aggregate_tps']=rtx.output_tokens_batch/(rtx.wall_ms_batch/1000)
rtx['sequence_tps']=rtx.output_tokens_per_request/(rtx.wall_ms_batch/1000)
rtx=rtx[(rtx.input_tokens==128)&(rtx.max_new_tokens==64)]
rg=rtx.groupby('batch_size').agg(n=('wall_ms_batch','size'),batch_ms=('wall_ms_batch','mean'),amortized_ms=('wall_ms_per_request','mean'),aggregate_tps=('aggregate_tps','mean'),sequence_tps=('sequence_tps','mean'),memory=('peak_torch_memory_mb','mean'))
rg.to_csv(TAB/'rtx_corrected_metrics.csv')
write('rtx.tex',[f'{int(b)} & {int(r.n)} & {r.batch_ms/1000:.3f} & {r.amortized_ms/1000:.3f} & {r.aggregate_tps:.2f} & {r.sequence_tps:.2f} & {r.memory:.1f}'+r' \\' for b,r in rg.iterrows()])
print('Generated figures and tables. Corrected RTX metrics:');print(rg.to_string())
print('Adaptive reset counts:');print(resets.to_string())
