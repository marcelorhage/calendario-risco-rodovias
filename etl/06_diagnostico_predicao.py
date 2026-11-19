"""
Etapa 6: Análises diagnósticas ("Por que aconteceu?"), testes de hipóteses e análise preditiva.

Entrada: saida/base_final.parquet, saida/inmet/*, Calendário de Feriados 2020–2026 - Feriados.csv
Saída:   saida/diagnostico/resultados.json, tabelas CSV e figuras PNG
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'saida', 'diagnostico')
FIG = os.path.join(BASE, 'saida', 'eda', 'figuras')
os.makedirs(OUT, exist_ok=True)
AZUL, LARANJA, CINZA, TINTA, TINTA2 = '#2a78d6', '#eb6834', '#b9b8b1', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e8e7e3', 'axes.axisbelow': True,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'axes.edgecolor': CINZA, 'xtick.color': TINTA2, 'ytick.color': TINTA2})
FONTE = 'Fonte: PRF, INMET, IBGE e calendário de feriados (2020–2025). Elaboração da equipe.'
R = {}


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


df = pd.read_parquet(os.path.join(BASE, 'saida', 'base_final.parquet'))

# ---------------------------------------------------------------- calendário diário (classificação nacional)
cal = pd.read_csv(os.path.join(BASE, 'Calendário de Feriados 2020–2026 - Feriados.csv'), encoding='utf-8-sig',
                  dtype=str)
cal['data'] = pd.to_datetime(cal['data'], format='%d/%m/%Y')
cal = cal[cal['grupo_evento'] != 'São João']
TIPO = {'Feriado nacional': 'Feriado', 'Ponto facultativo': 'Ponto facultativo',
        'Ponto facultativo (meio período)': 'Ponto facultativo', 'Período festivo': 'Período festivo',
        'Feriadão': 'Feriadão'}


def calendario(inicio, fim):
    d = pd.DataFrame({'data': pd.date_range(inicio, fim)})
    d = d.merge(cal[['data', 'tipo', 'grupo_evento']], on='data', how='left')
    d['tipo_dia'] = d['tipo'].map(TIPO)
    d['dow'] = d['data'].dt.dayofweek
    d.loc[d['tipo_dia'].isna() & (d['dow'] >= 5), 'tipo_dia'] = 'Fim de semana comum'
    d['tipo_dia'] = d['tipo_dia'].fillna('Dia útil comum')
    d['especial'] = ~d['tipo_dia'].isin(['Dia útil comum', 'Fim de semana comum'])
    d['grupo_evento'] = d['grupo_evento'].fillna('Nenhum')
    nuc = np.sort(cal.loc[cal['tipo'].isin(['Feriado nacional', 'Ponto facultativo']), 'data'].unique())
    pos = np.searchsorted(nuc, d['data'].values)
    ant = nuc[np.clip(pos - 1, 0, len(nuc) - 1)]
    prox = nuc[np.clip(pos, 0, len(nuc) - 1)]
    da = (d['data'].values - ant) / np.timedelta64(1, 'D')
    dp = (prox - d['data'].values) / np.timedelta64(1, 'D')
    dist = np.where(dp == 0, 0, np.where(da <= dp, da, -dp))
    d['janela'] = np.where(np.abs(dist) <= 3, dist, 99).astype(int).astype(str)
    d.loc[d['janela'] == '99', 'janela'] = 'fora'
    d['mes'] = d['data'].dt.month
    d['ano'] = d['data'].dt.year
    d['pandemia'] = d['data'].between('2020-03-11', '2021-12-31')
    d['tendencia'] = (d['data'] - pd.Timestamp('2020-01-01')).dt.days / 365.25
    return d


dia = calendario('2020-01-01', '2025-12-31')
agg = df.groupby('data').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'), fatais=('acidente_fatal', 'sum'))
dia = dia.merge(agg, left_on='data', right_index=True, how='left').fillna({'acidentes': 0, 'mortos': 0, 'fatais': 0})

# ================================================================ H1 - datas especiais x dias comuns
uteis_fer = dia[(dia['dow'] < 5) & dia['tipo_dia'].isin(['Feriado', 'Ponto facultativo'])]
uteis_com = dia[dia['tipo_dia'] == 'Dia útil comum']
fds_esp = dia[(dia['dow'] >= 5) & dia['especial']]
fds_com = dia[dia['tipo_dia'] == 'Fim de semana comum']
h1 = {}
for nome, a, b in [('feriado_em_dia_util_vs_dia_util_comum', uteis_fer, uteis_com),
                   ('fds_de_feriadao_vs_fds_comum', fds_esp, fds_com)]:
    for col in ['acidentes', 'mortos']:
        u = stats.mannwhitneyu(a[col], b[col], alternative='two-sided')
        h1[f'{nome}_{col}'] = {'media_grupo': r(a[col].mean(), 2), 'media_comparacao': r(b[col].mean(), 2),
                               'diferenca_%': r((a[col].mean() / b[col].mean() - 1) * 100, 1),
                               'n_dias': [len(a), len(b)], 'mann_whitney_p': r(u.pvalue, 6)}
# letalidade: qui-quadrado (acidente fatal x data especial)
tab = pd.crosstab(df['eh_data_especial'], df['acidente_fatal'])
chi = stats.chi2_contingency(tab)
h1['letalidade_especial_vs_comum'] = {
    'pct_fatais_especial': r(tab.loc[True, True] / tab.loc[True].sum() * 100, 2),
    'pct_fatais_comum': r(tab.loc[False, True] / tab.loc[False].sum() * 100, 2), 'qui2_p': r(chi[1], 8)}
# modelo binomial negativo: mortos por dia, controlando dia da semana, mês, ano e pandemia
nb = smf.glm('mortos ~ C(dow) + C(mes) + C(ano) + pandemia + especial', data=dia,
             family=sm.families.NegativeBinomial(alpha=0.05)).fit()
nba = smf.glm('acidentes ~ C(dow) + C(mes) + C(ano) + pandemia + especial', data=dia,
              family=sm.families.NegativeBinomial(alpha=0.02)).fit()
h1['modelo_mortos_dia'] = {'efeito_data_especial_%': r((np.exp(nb.params['especial[T.True]']) - 1) * 100, 1),
                           'ic95_%': [r((np.exp(x) - 1) * 100, 1) for x in nb.conf_int().loc['especial[T.True]']],
                           'p': r(nb.pvalues['especial[T.True]'], 6)}
h1['modelo_acidentes_dia'] = {'efeito_data_especial_%': r((np.exp(nba.params['especial[T.True]']) - 1) * 100, 1),
                              'ic95_%': [r((np.exp(x) - 1) * 100, 1) for x in nba.conf_int().loc['especial[T.True]']],
                              'p': r(nba.pvalues['especial[T.True]'], 6)}
R['H1'] = h1

# ================================================================ H2 - madrugadas de fim de semana
tab = pd.crosstab(df['madrugada_fds'], df['acidente_fatal'])
chi = stats.chi2_contingency(tab)
p1, p0 = tab.loc[True, True] / tab.loc[True].sum(), tab.loc[False, True] / tab.loc[False].sum()
mad = df[df['hora'] <= 5]
tab2 = pd.crosstab(mad['fim_de_semana'], mad['acidente_fatal'])
R['H2'] = {'pct_fatais_madrugada_fds': r(p1 * 100, 2), 'pct_fatais_demais': r(p0 * 100, 2),
           'risco_relativo': r(p1 / p0, 2), 'qui2_p': r(chi[1], 10),
           'letalidade_madrugada_fds': r(df.loc[df.madrugada_fds, 'mortos'].sum() / df.madrugada_fds.sum() * 100, 2),
           'letalidade_madrugada_dia_util': r(mad.loc[~mad.fim_de_semana, 'mortos'].sum()
                                              / (~mad.fim_de_semana).sum() * 100, 2),
           'pct_acidentes_madrugada_fds': r(df['madrugada_fds'].mean() * 100, 2),
           'pct_mortos_madrugada_fds': r(df.loc[df.madrugada_fds, 'mortos'].sum() / df['mortos'].sum() * 100, 2),
           'qui2_p_madrugada_fds_vs_madrugada_util': r(stats.chi2_contingency(tab2)[1], 6)}

# ================================================================ H3 - horários de pico
h = df.groupby('hora').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
h['letalidade'] = h['mortos'] / h['acidentes'] * 100
pico = h.index.isin([6, 7, 8, 17, 18, 19])
rho = stats.spearmanr(h['acidentes'], h['letalidade'])
R['H3'] = {'pct_acidentes_pico': r(h.loc[pico, 'acidentes'].sum() / h['acidentes'].sum() * 100, 1),
           'pct_horas_pico': r(6 / 24 * 100, 1),
           'letalidade_pico': r(h.loc[pico, 'mortos'].sum() / h.loc[pico, 'acidentes'].sum() * 100, 2),
           'letalidade_fora_pico': r(h.loc[~pico, 'mortos'].sum() / h.loc[~pico, 'acidentes'].sum() * 100, 2),
           'spearman_volume_letalidade': r(rho.statistic, 3), 'spearman_p': r(rho.pvalue, 6),
           'hora_mais_acidentes': int(h['acidentes'].idxmax()), 'hora_maior_letalidade': int(h['letalidade'].idxmax())}

# ================================================================ H4 - chuva (frequência e gravidade)
# Exposição: fração das horas com chuva nas estações usadas, ponderada pelo nº de acidentes de cada estação-ano.
fr = []
for f in sorted(glob.glob(os.path.join(BASE, 'saida', 'inmet', 'inmet_*.parquet'))):
    ano = int(os.path.basename(f)[6:10])
    m = pd.read_parquet(f, columns=['estacao', 'chuva_mm'])
    m['estacao'] = m['estacao'].astype(str)
    g = m.dropna().groupby('estacao')['chuva_mm'].apply(lambda s: (s > 0).mean()).rename('frac_chuva').reset_index()
    g['ano'] = ano
    fr.append(g)
fr = pd.concat(fr)
cl = df[df['tem_dado_clima']].copy()
peso = cl.groupby(['estacao_inmet', 'ano']).size().rename('n').reset_index().rename(columns={'estacao_inmet': 'estacao'})
peso = peso.merge(fr, on=['estacao', 'ano'], how='left')
exposicao = np.average(peso['frac_chuva'], weights=peso['n'])
obs = cl['choveu'].astype(bool).mean()
tab = pd.crosstab(cl['choveu'].astype(bool), cl['acidente_fatal'])
chi = stats.chi2_contingency(tab)
pc, ps = tab.loc[True, True] / tab.loc[True].sum(), tab.loc[False, True] / tab.loc[False].sum()
R['H4'] = {'pct_horas_com_chuva_nas_estacoes': r(exposicao * 100, 2), 'pct_acidentes_com_chuva': r(obs * 100, 2),
           'razao_frequencia': r(obs / exposicao, 2),
           'pct_fatais_com_chuva': r(pc * 100, 2), 'pct_fatais_sem_chuva': r(ps * 100, 2),
           'risco_relativo_fatal': r(pc / ps, 2), 'qui2_p': r(chi[1], 8),
           'letalidade_com_chuva': r(cl.loc[cl.choveu.astype(bool), 'mortos'].sum() / cl.choveu.astype(bool).sum() * 100, 2),
           'letalidade_sem_chuva': r(cl.loc[~cl.choveu.astype(bool), 'mortos'].sum() / (~cl.choveu.astype(bool)).sum() * 100, 2),
           'n_acidentes_com_clima': int(len(cl))}

# ================================================================ H5 - urbano x rural
tab = pd.crosstab(df['urbano'], df['acidente_fatal'])
chi = stats.chi2_contingency(tab)
R['H5'] = {'pct_acidentes_urbanos': r(df['urbano'].mean() * 100, 1),
           'letalidade_urbano': r(df.loc[df.urbano, 'mortos'].sum() / df.urbano.sum() * 100, 2),
           'letalidade_rural': r(df.loc[~df.urbano, 'mortos'].sum() / (~df.urbano).sum() * 100, 2),
           'pct_fatais_urbano': r(tab.loc[True, True] / tab.loc[True].sum() * 100, 2),
           'pct_fatais_rural': r(tab.loc[False, True] / tab.loc[False].sum() * 100, 2), 'qui2_p': r(chi[1], 10)}

# ================================================================ H6 - concentração em trechos (Pareto / Gini)
t = df.dropna(subset=['trecho']).groupby('trecho')['mortos'].sum().sort_values(ascending=False)
cum = t.cumsum() / t.sum()
x = np.sort(t.values)
gini = 1 - 2 * np.sum(np.cumsum(x) / x.sum()) / len(x) + 1 / len(x)
R['H6'] = {'trechos': int(len(t)), 'pct_trechos_para_50pct_mortes': r(((cum < .5).sum() + 1) / len(t) * 100, 1),
           'pct_trechos_para_80pct_mortes': r(((cum < .8).sum() + 1) / len(t) * 100, 1),
           'pct_mortes_nos_5pct_piores_trechos': r(t.head(int(len(t) * .05)).sum() / t.sum() * 100, 1),
           'gini_mortes_por_trecho': r(gini, 3)}

# ================================================================ H7 - tendência pós-pandemia
mes = dia.groupby(dia['data'].dt.to_period('M')).agg(acidentes=('acidentes', 'sum'), mortos=('mortos', 'sum'))
mes = mes[mes.index >= pd.Period('2022-01')]
xm = np.arange(len(mes))
la, lm = stats.linregress(xm, mes['acidentes']), stats.linregress(xm, mes['mortos'])
ano = df.groupby('ano').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
R['H7'] = {'acidentes_2021': int(ano.loc[2021, 'acidentes']), 'acidentes_2025': int(ano.loc[2025, 'acidentes']),
           'var_acidentes_2021_2025_%': r((ano.loc[2025, 'acidentes'] / ano.loc[2021, 'acidentes'] - 1) * 100, 1),
           'var_mortos_2021_2025_%': r((ano.loc[2025, 'mortos'] / ano.loc[2021, 'mortos'] - 1) * 100, 1),
           'tendencia_acidentes_por_ano_2022_2025': r(la.slope * 12, 1), 'p_acidentes': r(la.pvalue, 8),
           'tendencia_mortos_por_ano_2022_2025': r(lm.slope * 12, 1), 'p_mortos': r(lm.pvalue, 8)}

# ================================================================ DIAGNÓSTICO: regressão logística (acidente fatal)
m = df.copy()
m['fatal'] = m['acidente_fatal'].astype(int)
m['chuva'] = m['intensidade_chuva'].astype(str).replace({'nan': 'Sem dado'})
m['faixa'] = m['faixa_horaria'].astype(str)
m['dia_especial'] = m['eh_data_especial'].astype(int)
m['fds'] = m['fim_de_semana'].astype(int)
m['urb'] = m['urbano'].astype(int)
m['tipo_pista'] = m['tipo_pista'].fillna('Simples')
form = ("fatal ~ C(faixa, Treatment('Tarde (12–17h)')) + fds + dia_especial + urb + C(tipo_pista, Treatment('Dupla'))"
        " + C(grupo_causa, Treatment('Falta de atenção / reação do condutor')) + C(chuva, Treatment('Sem chuva'))"
        " + C(ano)")
logit = smf.logit(form, data=m).fit(disp=0, maxiter=200)
ors = pd.DataFrame({'odds_ratio': np.exp(logit.params), 'ic95_inf': np.exp(logit.conf_int()[0]),
                    'ic95_sup': np.exp(logit.conf_int()[1]), 'p': logit.pvalues})
ors = ors.drop(index=[i for i in ors.index if i.startswith('C(ano)') or i == 'Intercept'])


def limpa_nome(i):
    for a, b in [("C(faixa, Treatment('Tarde (12–17h)'))[T.", 'Horário: '), ("C(tipo_pista, Treatment('Dupla'))[T.", 'Pista: '),
                 ("C(grupo_causa, Treatment('Falta de atenção / reação do condutor'))[T.", 'Causa: '),
                 ("C(chuva, Treatment('Sem chuva'))[T.", 'Chuva: ')]:
        i = i.replace(a, b)
    return {'fds': 'Fim de semana', 'dia_especial': 'Data especial (feriado etc.)', 'urb': 'Trecho urbano'}.get(i, i.rstrip(']'))


ors.index = [limpa_nome(i) for i in ors.index]
ors = ors.sort_values('odds_ratio', ascending=False)
ors.round(4).to_csv(os.path.join(OUT, 'regressao_logistica_odds_ratios.csv'), encoding='utf-8')
R['logistica'] = {'n': int(logit.nobs), 'pseudo_r2': r(logit.prsquared, 4),
                  'referencias': 'Tarde (12–17h); pista dupla; causa = falta de atenção/reação; sem chuva; dia útil; rural',
                  'odds_ratios': ors.round(3).reset_index().rename(columns={'index': 'fator'}).to_dict('records')}

o = ors[~ors.index.str.contains('Sem dado')].sort_values('odds_ratio')
fig, ax = plt.subplots(figsize=(9, 7.5))
cores = [LARANJA if v > 1 else AZUL for v in o['odds_ratio']]
ax.hlines(range(len(o)), o['ic95_inf'], o['ic95_sup'], color=CINZA, lw=2)
ax.scatter(o['odds_ratio'], range(len(o)), color=cores, s=40, zorder=3, edgecolor='white', lw=1.5)
ax.axvline(1, color=TINTA2, lw=1, ls='--')
ax.set_xscale('log')
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10], ['0,2', '0,5', '1', '2', '5', '10'])
ax.set_yticks(range(len(o)), o.index, fontsize=8)
ax.grid(axis='y', visible=False)
ax.set(title='Fatores associados a um acidente ser fatal (razão de chances, IC 95%)',
       xlabel='Razão de chances (escala log) — >1 aumenta a chance de morte')
fig.text(0, -0.02, FONTE + ' Regressão logística, n = ' + f'{int(logit.nobs):,}'.replace(',', '.') + '.', fontsize=8,
         color=TINTA2)
fig.savefig(os.path.join(FIG, 'g11_regressao_logistica.png'))
plt.close(fig)

# ================================================================ CORRELAÇÕES
cl['chuva_bin'] = cl['choveu'].astype(bool)
diac = cl.groupby('data').agg(pct_acidentes_com_chuva=('chuva_bin', 'mean'), temperatura_media=('temperatura_c', 'mean'))
dc = dia.set_index('data')[['acidentes', 'mortos', 'dow', 'especial']].join(diac)
dc['letalidade'] = dc['mortos'] / dc['acidentes'] * 100
dc['fim_de_semana'] = (dc['dow'] >= 5).astype(int)
dc['data_especial'] = dc['especial'].astype(int)
corr = dc[['acidentes', 'mortos', 'letalidade', 'pct_acidentes_com_chuva', 'temperatura_media', 'fim_de_semana',
           'data_especial']].corr(method='spearman')
corr.round(3).to_csv(os.path.join(OUT, 'correlacao_spearman_diaria.csv'), encoding='utf-8')
R['correlacao_diaria_spearman'] = corr.round(3).to_dict()

uf = df.groupby('uf').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'), pct_rural=('urbano', lambda s: 1 - s.mean()),
                          pct_pista_simples=('tipo_pista', lambda s: (s == 'Simples').mean()))
pop = df.drop_duplicates(['ano', 'uf']).groupby('uf')['populacao_uf'].mean()
uf['acidentes_100mil'] = uf['acidentes'] / 6 / pop * 1e5
uf['mortos_100mil'] = uf['mortos'] / 6 / pop * 1e5
uf['letalidade'] = uf['mortos'] / uf['acidentes'] * 100
cu = {}
for a, b in [('pct_rural', 'letalidade'), ('pct_pista_simples', 'letalidade'), ('acidentes_100mil', 'mortos_100mil')]:
    s = stats.spearmanr(uf[a], uf[b])
    cu[f'{a}_x_{b}'] = {'rho': r(s.statistic, 3), 'p': r(s.pvalue, 5)}
R['correlacao_uf_spearman'] = cu

labels = ['Acidentes', 'Mortos', 'Letalidade', '% acid. c/ chuva', 'Temperatura', 'Fim de semana', 'Data especial']
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(7), labels, rotation=35, ha='right')
ax.set_yticks(range(7), labels)
ax.grid(False)
for (i, j), v in np.ndenumerate(corr.values):
    ax.text(j, i, f'{v:.2f}'.replace('.', ','), ha='center', va='center', fontsize=8,
            color='white' if abs(v) > .6 else TINTA)
fig.colorbar(im, ax=ax, fraction=.04)
ax.set_title('Correlação de Spearman entre indicadores diários')
fig.text(0, -0.02, FONTE, fontsize=8, color=TINTA2)
fig.savefig(os.path.join(FIG, 'g12_matriz_correlacao.png'))
plt.close(fig)

# ================================================================ CAUSALIDADE: Carnaval com e sem festa (experimento natural)
# Índice = acidentes nos dias de Carnaval / média dos mesmos dias da semana nas 4 semanas anteriores e posteriores
car = cal[cal['grupo_evento'] == 'Carnaval'].copy()
car['ano'] = car['data'].dt.year
res = []
serie = dia.set_index('data')
for a, g in car[car['ano'] <= 2025].groupby('ano'):
    d0, d1 = g['data'].min(), g['data'].max()
    viz = serie[((serie.index >= d0 - pd.Timedelta(days=28)) & (serie.index < d0))
                | ((serie.index > d1) & (serie.index <= d1 + pd.Timedelta(days=28)))]
    viz = viz[~viz['especial']]
    base_a = viz.groupby('dow')['acidentes'].mean()
    base_m = viz.groupby('dow')['mortos'].mean()
    s = serie.loc[g['data']]
    res.append({'ano': a, 'festa': a not in (2021, 2022), 'acidentes': int(s['acidentes'].sum()),
                'esperado_acidentes': r(s['dow'].map(base_a).sum(), 1),
                'indice_acidentes': r(s['acidentes'].sum() / s['dow'].map(base_a).sum() * 100, 1),
                'mortos': int(s['mortos'].sum()), 'esperado_mortos': r(s['dow'].map(base_m).sum(), 1),
                'indice_mortos': r(s['mortos'].sum() / s['dow'].map(base_m).sum() * 100, 1)})
car = pd.DataFrame(res)
com, sem = car[car.festa], car[~car.festa]
R['causalidade_carnaval'] = {'por_ano': car.to_dict('records'),
                             'indice_acidentes_com_festa': r(com['indice_acidentes'].mean(), 1),
                             'indice_acidentes_sem_festa': r(sem['indice_acidentes'].mean(), 1),
                             'indice_mortos_com_festa': r(com['indice_mortos'].mean(), 1),
                             'indice_mortos_sem_festa': r(sem['indice_mortos'].mean(), 1)}
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.axhline(100, color=TINTA2, lw=1, ls='--')
ax.bar(car['ano'].astype(str), car['indice_acidentes'], color=[AZUL if f else CINZA for f in car['festa']], width=.6)
for i, (v, f) in enumerate(zip(car['indice_acidentes'], car['festa'])):
    ax.text(i, v + 1, f"{v:.0f}\n{'com festa' if f else 'sem festa'}", ha='center', fontsize=8)
ax.set_ylim(0, car['indice_acidentes'].max() * 1.25)
ax.set(title='Acidentes no Carnaval (100 = dias semelhantes das semanas vizinhas)', ylabel='Índice')
fig.text(0, -0.02, FONTE + ' 2021 e 2022: Carnaval de rua cancelado (pandemia).', fontsize=8, color=TINTA2)
fig.savefig(os.path.join(FIG, 'g13_carnaval_experimento.png'))
plt.close(fig)

# ================================================================ PREDITIVA: binomial negativa (acidentes e mortos por dia)
fp = 'C(dow) + C(mes) + C(tipo_dia) + C(janela) + pandemia + tendencia'
treino, teste = dia[dia['ano'] <= 2024], dia[dia['ano'] == 2025]
pred = {}
for alvo, alpha in [('acidentes', 0.02), ('mortos', 0.05)]:
    mod = smf.glm(f'{alvo} ~ {fp}', data=treino, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    p = mod.predict(teste)
    ing = teste.merge(treino[treino.ano == 2024].groupby('dow')[alvo].mean().rename('ing'), left_on='dow',
                      right_index=True)['ing']
    pred[alvo] = {'mae_modelo': r(np.mean(np.abs(teste[alvo] - p)), 2),
                  'mape_modelo_%': r(np.mean(np.abs(teste[alvo] - p) / teste[alvo].clip(lower=1)) * 100, 1),
                  'mae_ingenuo_media_2024_por_dia_semana': r(np.mean(np.abs(teste[alvo] - ing)), 2),
                  'total_2025_real': int(teste[alvo].sum()), 'total_2025_previsto': r(p.sum(), 0)}
    if alvo == 'acidentes':
        val = teste[['data', alvo]].assign(previsto=p.values)
R['preditiva_validacao_2025'] = pred

fig, ax = plt.subplots(figsize=(10, 3.8))
vs = val.set_index('data').rolling(7).mean()
ax.plot(vs.index, vs['acidentes'], color=AZUL, lw=2, label='Real (média móvel 7 dias)')
ax.plot(vs.index, vs['previsto'], color=LARANJA, lw=2, ls='--', label='Previsto pelo modelo (média móvel 7 dias)')
ax.legend(frameon=False, loc='upper left', fontsize=9)
ax.set(title='Validação do modelo: acidentes por dia em 2025 (treino 2020–2024)', ylabel='Acidentes por dia')
fig.text(0, -0.02, FONTE, fontsize=8, color=TINTA2)
fig.savefig(os.path.join(FIG, 'g14_validacao_preditiva.png'))
plt.close(fig)

# previsão 2026 (modelo reestimado com 2020-2025)
fut = calendario('2026-01-01', '2026-12-31')
prev = fut[['data', 'tipo_dia', 'grupo_evento', 'dow']].copy()
for alvo, alpha in [('acidentes', 0.02), ('mortos', 0.05)]:
    mod = smf.glm(f'{alvo} ~ {fp}', data=dia, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    sf = mod.get_prediction(fut).summary_frame(alpha=0.05)
    prev[f'{alvo}_previsto'] = sf['mean'].round(1).values
    prev[f'{alvo}_ic95_inf'] = sf['mean_ci_lower'].round(1).values
    prev[f'{alvo}_ic95_sup'] = sf['mean_ci_upper'].round(1).values
prev.to_csv(os.path.join(OUT, 'previsao_2026_diaria.csv'), index=False, encoding='utf-8')
fer26 = (prev[prev['grupo_evento'] != 'Nenhum'].groupby('grupo_evento')
         .agg(inicio=('data', 'min'), fim=('data', 'max'), dias=('data', 'size'),
              acidentes_previstos=('acidentes_previsto', 'sum'), mortos_previstos=('mortos_previsto', 'sum'))
         .sort_values('inicio'))
fer26['mortos_por_dia'] = fer26['mortos_previstos'] / fer26['dias']
fer26.round(1).to_csv(os.path.join(OUT, 'previsao_2026_feriados.csv'), encoding='utf-8')
R['previsao_2026'] = {'acidentes_ano': r(prev['acidentes_previsto'].sum(), 0), 'mortos_ano': r(prev['mortos_previsto'].sum(), 0),
                      'feriados': fer26.round(1).reset_index().astype({'inicio': str, 'fim': str}).to_dict('records')}

with open(os.path.join(OUT, 'resultados.json'), 'w', encoding='utf-8') as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)
for k in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'causalidade_carnaval', 'preditiva_validacao_2025']:
    print(k, json.dumps(R[k], ensure_ascii=False, default=str))
print('logit pseudo R2', R['logistica']['pseudo_r2'])
print(ors.round(3).to_string())
print(corr.round(2).to_string())
print(cu)
print(fer26.round(1).to_string())
print('2026 total', R['previsao_2026']['acidentes_ano'], R['previsao_2026']['mortos_ano'])
