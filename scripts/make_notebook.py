"""Build analysis.ipynb from the markdown in notebook_text.py + code cells below.
Then run: jupyter nbconvert --to notebook --execute --inplace analysis.ipynb"""
import _bootstrap  # noqa: F401

import nbformat as nbf

from notebook_text import MD
from src.config import ROOT

C = {}
C["setup"] = """import sys, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
from src import market, pipeline, trend, bias, granger
from src.config import WINDOW_START, WINDOW_END, TABLES, FIGURES, OUTPUTS
src = pipeline.load_sources()
R = market.build_returns()
st = pipeline.streams(src)
idx = pipeline.build_indices(st, R)
res = json.loads((OUTPUTS / 'results.json').read_text())
{k: len(v) for k, v in src.items()}"""
C["counts"] = """display(pd.read_csv(TABLES / 'corpus_counts.csv', index_col=0))
wf = TABLES / 'twitter_filter_waterfall.csv'
if wf.exists():
    print('Tweet signal-to-noise filters, applied in order:')
    display(pd.read_csv(wf))
else:
    print('Twitter not collected yet: the filter waterfall appears here after 05_clean_and_score.py twitter')"""
C["validation"] = """try:
    display(pd.read_csv(TABLES / 'validation_scores.csv').round(3))
    display(pd.read_csv(TABLES / 'tool_agreement_kappa.csv', index_col=0).round(2))
except FileNotFoundError:
    print('validation labels not scored yet')"""
C["basket"] = """hw = market.in_window(R['hourly']); dw = market.in_window(R['daily'])
display(pd.Series(R['betas']).round(2).to_frame('beta vs SPY (Mar to Aug daily)').T)
display(dw[['EW', 'CW', 'SPY', 'AR_EW', 'sub_compute', 'sub_platforms', 'sub_anthropic_exposed']].round(2))
display(market.descriptives(hw, ['EW', 'CW', 'SPY', 'AR_EW', 'sub_compute', 'sub_platforms']).round(3))
pd.DataFrame({c: market.efficiency_tests(hw[c]) for c in ['EW', 'AR_EW', 'SPY', 'sub_compute']}).T.round(4)"""
C["trend"] = """display(pd.read_csv(TABLES / 'q1_trend_tests_net.csv', index_col=0).round(4))
display(pd.read_csv(TABLES / 'q1_trend_tests_neg_share.csv', index_col=0).round(4))
print('keyness by stream:', json.dumps(res.get('keyness_by_stream', {}), indent=1))
for f in ['fig1_sentiment_trend.png', 'figE_sentiment_by_source.png', 'figB_thelwall_event.png']:
    display(Image(filename=str(FIGURES / f), width=900))"""
C["events"] = """ev = pd.read_csv(TABLES / 'q1_event_windows.csv', index_col=0)
ev.pivot_table(index='event', columns='stream', values='change', sort=False).round(3)"""
C["spike"] = """display(pd.read_csv(TABLES / 'q1_spike_test.csv', index_col=0).round(4))
print('most negative days (main stream):', res['keyness_worst_days']['days'])
print('keyness by stream:'); display(pd.DataFrame(res['keyness_by_stream']).T)
print('rank of Sept 9 (1 = most negative day of the window):', res['sept9_rank'])
pd.read_csv(TABLES / 'q1_keyness_worst_days.csv', index_col=0).head(12).round(4)"""
C["stationarity"] = """ret = market.in_window(R['hourly'])['AR_EW']
rows = {'AR_EW (hourly)': granger.stationarity(ret)}
for k, d in idx['hourly'].items():
    s = d['net'].reindex(ret.index).interpolate(limit_direction='both')
    rows[f'{k} level'] = granger.stationarity(s)
    rows[f'{k} diff'] = granger.stationarity(s.diff())
pd.DataFrame(rows).T"""
C["granger"] = """gh = pd.read_csv(TABLES / 'q2_granger_hourly.csv', index_col=0)
display(gh[['test', 'direction', 'lag', 'N', 'F', 'p', 'p_hac', 'p_perm', 'p_fdr', 'sum_coef']].round(4))
print('leave-one-day-out, Reddit lag 3:'); display(pd.read_csv(TABLES / 'q2_leave_one_day_out_reddit_lag3.csv')[['dropped_day', 'N', 'F', 'p', 'sum_coef']].round(4))
gd = pd.read_csv(TABLES / 'q2_granger_daily.csv', index_col=0)
gd[['test', 'direction', 'lag', 'N', 'F', 'p', 'p_perm', 'sum_coef']].round(4)"""
C["var"] = """pd.DataFrame({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
              for k, v in res['var'].items()}).T.round(4)"""
C["heat"] = """display(pd.read_csv(TABLES / 'q2_per_ticker_pvalues.csv', index_col=0).round(3))
for f in ['fig2_granger.png', 'figC_irf_heatmap.png']:
    display(Image(filename=str(FIGURES / f), width=900))"""
C["robust"] = """rob = pd.read_csv(TABLES / 'q2_granger_robustness.csv', index_col=0)
print(json.dumps({k: v for k, v in res['robustness_summary'].items() if k != 'significant'}, indent=1))
pd.DataFrame(res['robustness_summary']['significant'])"""
C["bias"] = """display(pd.read_csv(TABLES / 'q3_group_negativity.csv', index_col=0).round(3))
try:
    display(pd.read_csv(TABLES / 'q3_placebo_negativity.csv', index_col=0).round(3))
except FileNotFoundError:
    pass
for f in ['fig3_bias.png', 'fig4_sacerdote_fig1_fig5.png', 'figF_bias_two_measures.png']:
    display(Image(filename=str(FIGURES / f), width=900))"""
C["lpm"] = """for k, v in res['lpm'].items():
    print(k, 'N =', v['N'], 'R2 =', round(v['R2'], 3), 'omitted (Intl major) mean =', round(v['omitted_mean'], 3))
    display(pd.DataFrame(v['coef']).T.round(4))"""
C["bias_more"] = """print('tone vs raw basket:', json.dumps(res['tone_vs_market'], indent=1))
print('tone vs abnormal (AI-specific) basket return:', json.dumps(res['tone_vs_market_ar'], indent=1))
display(pd.read_csv(TABLES / 'q3_frame_counts.csv', index_col=0).round(3))
print('S-1 framing:'); display(pd.DataFrame(res['s1_framing']).T)
print('base rate framing:', res['base_rate'])
print('ideology split:'); display(pd.DataFrame(res['ideology']).T)"""
C["demand"] = """for k in ['demand_twitter', 'demand_reddit']:
    if k in res:
        print(k); display(pd.DataFrame(res[k]).T)"""
C["e1"] = """gh[gh['test'].isin(['tw_RISK', 'tw_FIN', 'tw_GEN'])].pivot_table(
    index=['test', 'lag'], columns='direction', values='p').round(4)"""
C["e2"] = """r = rob[(rob['ret_col'] == 'AR_EW') & (rob['measure'] == 'n') & (rob['diff'] == True)]
r[['test', 'direction', 'lag', 'F', 'p', 'sum_coef']].round(4)"""
C["e3"] = """e3 = pd.read_csv(TABLES / 'e3_vol_windows.csv', index_col=0)
display(e3.pivot_table(index='window', columns='series', values='hourly_sd', sort=False).round(3))
display(e3.groupby('window', sort=False)[['amihud_ew', 'vix_mean']].first().round(4))
display(Image(filename=str(FIGURES / 'figA_aloosh_volatility.png'), width=900))
gs = pd.read_csv(TABLES / 'e3_granger_subbaskets.csv', index_col=0)
gs[gs['direction'] == 'sent -> ret'].pivot_table(index=['test', 'lag'], columns='ret_col', values='p').round(3)"""

order = ["intro", "setup", "data", "counts", "validation_md", "validation", "why_validation", "scorer_md",
         "basket_md", "basket", "why_basket",
         "trend_md", "trend", "how_trend_figs", "why_trend", "events_md", "events", "why_events", "spike_md", "spike", "why_spike",
         "granger_md", "stationarity", "granger", "why_granger", "var_md", "var", "heat", "how_granger_figs", "why_var",
         "robust_md", "robust", "why_robust", "bias_md", "bias", "how_bias_figs", "why_bias", "lpm_md", "lpm", "why_lpm",
         "bias_more_md", "bias_more", "why_bias_more", "demand_md", "demand", "why_demand",
         "extra_md", "e1", "why_e1", "e2_md", "e2", "why_e2", "e3_md", "e3", "how_vol_fig", "why_e3", "wrap"]
nb = nbf.v4.new_notebook()
for k in order:
    if k in C:
        nb.cells.append(nbf.v4.new_code_cell(C[k]))
    else:
        nb.cells.append(nbf.v4.new_markdown_cell(MD[k]))
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "analysis.ipynb")
print("wrote analysis.ipynb with", len(nb.cells), "cells")
