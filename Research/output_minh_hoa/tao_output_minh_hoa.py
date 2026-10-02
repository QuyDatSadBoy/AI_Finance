"""Two real strategies; only EB thetahat is invented. Never fits EB."""
import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FAMILY = 'DataMinedLongShortReturnsEW'
FAKE_THETA = {66: -2.5, 67: -1.5}

def save(name, fields, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

monthly = []
with (ROOT / 'data_raw/DataMinedLongShortReturnsEW.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        sid = int(row['signalid'])
        if sid not in FAKE_THETA:
            continue
        year = int(row['year'])
        if 2000 <= year <= 2020:
            monthly.append(dict(signalid=sid, date=f"{year}-{int(row['month']):02d}-01",
                                year=year, ret=float(row['ret'])))
monthly.sort(key=lambda r: (r['signalid'], r['date']))
save('MINH_HOA_signal_dic.csv', ['signalid', 'date', 'year', 'ret'], monthly)

fields = ['signal_family', 'oos_begin_year', 'signalid', 'mean_ret', 'tstat',
          'nobs', 's_flag', 'is_begin_year']
stats = []
for sid in FAKE_THETA:
    for flag in ('IS', 'OOS'):
        values = [r['ret'] for r in monthly if r['signalid'] == sid
                  and ((r['year'] < 2020) if flag == 'IS' else (r['year'] == 2020))]
        mean = statistics.mean(values)
        # Intercept-only OLS, HAC maxlags=0, no finite-sample correction.
        se = math.sqrt(sum((v - mean) ** 2 for v in values)) / len(values)
        stats.append(dict(zip(fields, [FAMILY, 2020, sid, mean, mean / se,
                                       len(values), flag, 2000])))
save('OOS_signal_tstat_OosNyears1.csv', fields, stats)
predictions = []
for row in stats:
    if row['s_flag'] == 'IS':
        theta = FAKE_THETA[row['signalid']]
        predictions.append({**row, 'thetahat': theta,
                            'pred_ret': theta * row['mean_ret'] / row['tstat']})
save('MINH_HOA_Predict_SignalYear.csv', fields + ['thetahat', 'pred_ret'], predictions)

signed = []
for row in predictions:
    for month in monthly:
        if month['signalid'] != row['signalid'] or month['year'] != 2020:
            continue
        sign = 1 if row['pred_ret'] >= 0 else -1
        sign_tstat = 1 if row['tstat'] >= 0 else -1
        signed.append(dict(signal_family='acct_ew', signalid=row['signalid'],
                           date=month['date'], pred_ret=sign * row['pred_ret'],
                           ret_oos=sign * month['ret'], thetahat=sign * row['thetahat'],
                           tstat_is=sign_tstat * row['tstat'], nobs=row['nobs'],
                           oos_begin_year=2020, sign_tstat=sign_tstat,
                           ret_oos_tstat=sign_tstat * month['ret'],
                           rank_pct_tstat=0, sign=sign,
                           SRahat=sign * row['thetahat'] / math.sqrt(row['nobs']) * math.sqrt(12),
                           rank_pct=0))
for date in sorted({r['date'] for r in signed}):
    group = [r for r in signed if r['date'] == date]
    for score, target in [('tstat_is', 'rank_pct_tstat'), ('SRahat', 'rank_pct')]:
        for row in group:
            smaller = sum(r[score] < row[score] for r in group)
            ties = sum(r[score] == row[score] for r in group)
            row[target] = (smaller + (ties + 1) / 2) / len(group) * 100
save('MINH_HOA_df_signed.csv', list(signed[0]), signed)

best = []
for sort_var, rank_key, ret_key in [('SRahat', 'rank_pct', 'ret_oos'),
                                    ('tstat_is', 'rank_pct_tstat', 'ret_oos_tstat')]:
    for cutoff in (1, 5, 10):
        for date in sorted({r['date'] for r in signed}):
            selected = [r for r in signed if r['date'] == date and r[rank_key] >= 100-cutoff]
            best.append(dict(date=date, ret_oos=statistics.mean(r[ret_key] for r in selected),
                             nstrat=len(selected), pctmin=cutoff, sort_var=sort_var))
save('MINH_HOA_df_best_ret.csv', ['date', 'ret_oos', 'nstrat', 'pctmin', 'sort_var'], best)
values = [r['ret_oos'] for r in best if r['sort_var'] == 'SRahat' and r['pctmin'] == 1]
mean, sd = statistics.mean(values), statistics.stdev(values)
save('MINH_HOA_tabdat.csv', ['name', 'nstrat_mean', 'ret_mean', 'tstat', 'sr'],
     [dict(name='EB Mining Top 1%', nstrat_mean=1, ret_mean=mean * 12,
           tstat=mean / sd * math.sqrt(len(values)), sr=mean / sd * math.sqrt(12))])
print('Created six CSVs; only thetahat was invented. Rankings use the two-signal example.')
