"""
Etapa 5: Análise Exploratória de Dados (EDA) - "O que aconteceu?"

Entrada: saida/base_final.parquet
Saída:   saida/eda/estatistica_descritiva.csv   (média, mediana, quartis, desvio padrão)
         saida/eda/tabelas/*.csv               (tabelas de apoio)
         saida/eda/figuras/*.png               (gráficos do relatório)
         saida/eda/resumo_eda.json             (números citados no relatório)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'saida', 'eda')
FIG = os.path.join(OUT, 'figuras')
TAB = os.path.join(OUT, 'tabelas')
for p in (OUT, FIG, TAB):
    os.makedirs(p, exist_ok=True)

# ---------------------------------------------------------------- estilo (paleta validada)
AZUL, LARANJA, CINZA, TINTA, TINTA2 = '#2a78d6', '#eb6834', '#b9b8b1', '#0b0b0b', '#52514e'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': CINZA, 'axes.labelcolor': TINTA2,
    'xtick.color': TINTA2, 'ytick.color': TINTA2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e8e7e3', 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.titleweight': 'bold', 'axes.titlesize': 12, 'axes.titlecolor': TINTA, 'axes.titlelocation': 'left',
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'figure.facecolor': 'white'})


def salva(fig, nome, fonte='Fonte: PRF, INMET, IBGE e calendário de feriados (2020–2025). Elaboração da equipe.'):
    fig.text(0.0, -0.02, fonte, fontsize=8, color=TINTA2, ha='left', va='top', transform=fig.transFigure)
    fig.savefig(os.path.join(FIG, nome))
    plt.close(fig)


def br(x, casas=0):
    s = f'{x:,.{casas}f}'
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


df = pd.read_parquet(os.path.join(BASE, 'saida', 'base_final.parquet'))
R = {}

# ================================================================ 1. ESTATÍSTICA DESCRITIVA
dia = df.groupby('data').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'),
                             feridos_graves=('feridos_graves', 'sum'), feridos=('feridos', 'sum'))
dia = dia.reindex(pd.date_range('2020-01-01', '2025-12-31'), fill_value=0)
dia.index.name = 'data'
dia['letalidade'] = dia['mortos'] / dia['acidentes'] * 100

variaveis = {
    'Acidentes por dia': dia['acidentes'],
    'Mortos por dia': dia['mortos'],
    'Feridos graves por dia': dia['feridos_graves'],
    'Letalidade diária (mortos/100 acidentes)': dia['letalidade'],
    'Pessoas envolvidas por acidente': df['pessoas'],
    'Veículos por acidente': df['veiculos'],
    'Mortos por acidente': df['mortos'],
    'Feridos por acidente': df['feridos'],
    'Chuva na hora do acidente (mm)': df['chuva_mm'],
    'Temperatura na hora do acidente (°C)': df['temperatura_c'],
    'Distância até a estação do INMET (km)': df['distancia_estacao_km'],
}
linhas = []
for nome, s in variaveis.items():
    s = s.dropna()
    q = s.quantile([.25, .5, .75])
    linhas.append({'variavel': nome, 'n': len(s), 'media': s.mean(), 'mediana': q[.5], 'q1': q[.25], 'q3': q[.75],
                   'minimo': s.min(), 'maximo': s.max(), 'desvio_padrao': s.std(),
                   'coef_variacao_%': s.std() / s.mean() * 100 if s.mean() else np.nan,
                   'assimetria': s.skew()})
desc = pd.DataFrame(linhas)
desc.round(3).to_csv(os.path.join(OUT, 'estatistica_descritiva.csv'), index=False, encoding='utf-8')
R['descritiva'] = desc.round(3).to_dict('records')
R['dia_recorde'] = {'data': str(dia['acidentes'].idxmax().date()), 'acidentes': int(dia['acidentes'].max())}
R['dia_mais_mortes'] = {'data': str(dia['mortos'].idxmax().date()), 'mortos': int(dia['mortos'].max())}

# ================================================================ 2. TABELAS DE APOIO
ano = df.groupby('ano').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'), feridos_graves=('feridos_graves', 'sum'),
                            acidentes_fatais=('acidente_fatal', 'sum'))
pop_ano = df.drop_duplicates(['ano', 'uf']).groupby('ano')['populacao_uf'].sum()
ano['letalidade'] = ano['mortos'] / ano['acidentes'] * 100
ano['mortos_100mil'] = ano['mortos'] / pop_ano * 1e5
ano['var_acidentes_%'] = ano['acidentes'].pct_change() * 100
ano.round(2).to_csv(os.path.join(TAB, 'por_ano.csv'), encoding='utf-8')
R['por_ano'] = ano.round(2).reset_index().to_dict('records')

dias_tipo = df.drop_duplicates('data')[['data', 'tipo_dia']]
# o tipo de dia do São João varia por UF; para a contagem de dias usamos a classificação nacional
td = df.groupby('tipo_dia').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
nd = df.groupby('tipo_dia')['data'].nunique()
td['dias'] = nd
td['acidentes_por_dia'] = td['acidentes'] / td['dias']
td['mortos_por_dia'] = td['mortos'] / td['dias']
td['letalidade'] = td['mortos'] / td['acidentes'] * 100
td = td.sort_values('mortos_por_dia', ascending=False)
td.round(2).to_csv(os.path.join(TAB, 'por_tipo_dia.csv'), encoding='utf-8')
R['por_tipo_dia'] = td.round(2).reset_index().to_dict('records')

ev = df[df['grupo_evento'] != 'Nenhum'].groupby('grupo_evento').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'),
                                                                   dias=('data', 'nunique'))
ev['acidentes_por_dia'] = ev['acidentes'] / ev['dias']
ev['mortos_por_dia'] = ev['mortos'] / ev['dias']
ev['letalidade'] = ev['mortos'] / ev['acidentes'] * 100
ev = ev.sort_values('mortos_por_dia', ascending=False)
ev.round(2).to_csv(os.path.join(TAB, 'por_evento.csv'), encoding='utf-8')
R['por_evento'] = ev.round(2).reset_index().to_dict('records')

hora = df.groupby('hora').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
hora['letalidade'] = hora['mortos'] / hora['acidentes'] * 100
hora.round(2).to_csv(os.path.join(TAB, 'por_hora.csv'), encoding='utf-8')
R['por_hora'] = hora.round(2).reset_index().to_dict('records')

causa = df.groupby('grupo_causa').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
causa['letalidade'] = causa['mortos'] / causa['acidentes'] * 100
causa['%_acidentes'] = causa['acidentes'] / len(df) * 100
causa['%_mortos'] = causa['mortos'] / df['mortos'].sum() * 100
causa = causa.sort_values('mortos', ascending=False)
causa.round(2).to_csv(os.path.join(TAB, 'por_grupo_causa.csv'), encoding='utf-8')
R['por_grupo_causa'] = causa.round(2).reset_index().to_dict('records')

tipo = df.groupby('tipo_acidente_padronizado').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
tipo['letalidade'] = tipo['mortos'] / tipo['acidentes'] * 100
tipo = tipo.sort_values('mortos', ascending=False)
tipo.round(2).to_csv(os.path.join(TAB, 'por_tipo_acidente.csv'), encoding='utf-8')
R['por_tipo_acidente'] = tipo.round(2).reset_index().to_dict('records')

pop_uf = df.drop_duplicates(['ano', 'uf']).groupby('uf')['populacao_uf'].mean()
uf = df.groupby('uf').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
uf['letalidade'] = uf['mortos'] / uf['acidentes'] * 100
uf['acidentes_100mil_ano'] = uf['acidentes'] / 6 / pop_uf * 1e5
uf['mortos_100mil_ano'] = uf['mortos'] / 6 / pop_uf * 1e5
uf = uf.sort_values('mortos_100mil_ano', ascending=False)
uf.round(2).to_csv(os.path.join(TAB, 'por_uf.csv'), encoding='utf-8')
R['por_uf'] = uf.round(2).reset_index().to_dict('records')

urb = df.groupby('urbano').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
urb['letalidade'] = urb['mortos'] / urb['acidentes'] * 100
R['urbano_rural'] = urb.round(2).reset_index().to_dict('records')

fase = df.groupby('fase_dia').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
fase['letalidade'] = fase['mortos'] / fase['acidentes'] * 100
R['fase_dia'] = fase.round(2).reset_index().to_dict('records')

pista = df.groupby('tipo_pista').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
pista['letalidade'] = pista['mortos'] / pista['acidentes'] * 100
R['tipo_pista'] = pista.round(2).reset_index().to_dict('records')

trecho = df.dropna(subset=['trecho']).groupby('trecho').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'),
                                                           municipio=('municipio', lambda s: s.mode().iat[0]))
trecho = trecho.sort_values('mortos', ascending=False)
trecho.head(50).to_csv(os.path.join(TAB, 'top50_trechos_mortos.csv'), encoding='utf-8')
cum = trecho['mortos'].cumsum() / trecho['mortos'].sum()
R['trechos'] = {'total_trechos': int(len(trecho)),
                'trechos_com_morte': int((trecho['mortos'] > 0).sum()),
                'pct_trechos_50pct_mortes': round(float(((cum <= 0.5).sum() + 1) / len(trecho) * 100), 2),
                'n_trechos_50pct_mortes': int((cum <= 0.5).sum() + 1),
                'pct_mortes_top_1pct': round(float(trecho['mortos'].head(int(len(trecho) * .01)).sum()
                                                   / trecho['mortos'].sum() * 100), 2),
                'top10': trecho.head(10).reset_index().to_dict('records')}

clima = df[df['tem_dado_clima']].groupby('intensidade_chuva', observed=True).agg(acidentes=('id', 'size'),
                                                                                mortos=('mortos', 'sum'))
clima['letalidade'] = clima['mortos'] / clima['acidentes'] * 100
R['clima'] = clima.round(2).reset_index().astype({'intensidade_chuva': str}).to_dict('records')

# ================================================================ 3. FIGURAS
nomes_dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']

# G1 - histograma dos acidentes por dia
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(dia['acidentes'], bins=50, color=AZUL, edgecolor='white', linewidth=1)
for v, rot, c, ha in [(dia['acidentes'].mean(), 'média', TINTA, 'left'),
                      (dia['acidentes'].median(), 'mediana', LARANJA, 'right')]:
    ax.axvline(v, color=c, lw=1.5, ls='--')
    ax.text(v + (3 if ha == 'left' else -3), ax.get_ylim()[1] * .95, f'{rot}: {br(v, 1)}', color=c, fontsize=9,
            va='top', ha=ha, bbox=dict(fc='white', ec='none', pad=1.5))
ax.set(title='Distribuição do número de acidentes por dia (2020–2025)', xlabel='Acidentes no dia',
       ylabel='Número de dias')
salva(fig, 'g01_histograma_acidentes_dia.png')

# G2 - boxplot acidentes e mortos por dia, por tipo de dia
ordem = ['Dia útil comum', 'Ponto facultativo', 'Feriado', 'Período festivo', 'Feriadão', 'Fim de semana comum']
dd = df.groupby(['data', 'tipo_dia']).agg(acidentes=('id', 'size'), mortos=('mortos', 'sum')).reset_index()
dd = dd.sort_values('acidentes').drop_duplicates('data', keep='last')  # 1 tipo por dia (o predominante)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, col, tit in [(axs[0], 'acidentes', 'Acidentes por dia'), (axs[1], 'mortos', 'Mortos por dia')]:
    dados = [dd.loc[dd['tipo_dia'] == t, col] for t in ordem]
    bp = ax.boxplot(dados, vert=False, patch_artist=True, widths=.55, medianprops={'color': TINTA, 'lw': 1.5},
                    flierprops={'marker': 'o', 'ms': 3, 'mfc': CINZA, 'mec': 'none'}, whiskerprops={'color': CINZA},
                    capprops={'color': CINZA})
    for p, t in zip(bp['boxes'], ordem):
        p.set(facecolor=LARANJA if t == 'Fim de semana comum' else AZUL, edgecolor='white', alpha=.9)
    ax.set_yticks(range(1, len(ordem) + 1), ordem if col == 'acidentes' else [''] * len(ordem))
    ax.set_title(tit)
fig.suptitle('Distribuição diária por tipo de dia', x=0.01, ha='left', fontweight='bold', fontsize=12, y=1.02)
salva(fig, 'g02_boxplot_tipo_dia.png')

# G3 - série mensal
mes = df.groupby(pd.Grouper(key='data', freq='MS')).agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'))
fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for ax, col, tit in [(axs[0], 'acidentes', 'Acidentes por mês'), (axs[1], 'mortos', 'Mortos por mês')]:
    ax.axvspan(pd.Timestamp('2020-03-11'), pd.Timestamp('2021-12-31'), color='#f0efec', zorder=0)
    ax.plot(mes.index, mes[col], color=AZUL, lw=2)
    ax.plot(mes.index, mes[col].rolling(12, center=True).mean(), color=TINTA2, lw=1.2, ls='--')
    ax.set_title(tit)
axs[0].text(pd.Timestamp('2020-04-15'), axs[0].get_ylim()[1] * .98, 'período da pandemia', fontsize=8,
            color=TINTA2, va='top')
axs[1].text(mes.index[-1], mes['mortos'].rolling(12, center=True).mean().dropna().iloc[-1],
            ' média móvel 12 meses', fontsize=8, color=TINTA2, va='center')
fig.suptitle('Evolução mensal dos acidentes e mortes (2020–2025)', x=0.01, ha='left', fontweight='bold', fontsize=12)
salva(fig, 'g03_serie_mensal.png')

# G4 - mapa de calor dia da semana x hora (média de acidentes por dia, e letalidade)
cont = df.groupby(['dia_semana_num', 'hora']).size().unstack()
ndias = dia.groupby(dia.index.dayofweek).size()
cont = cont.div(ndias, axis=0)
let = df.groupby(['dia_semana_num', 'hora']).apply(lambda g: g['mortos'].sum() / len(g) * 100,
                                                    include_groups=False).unstack()
fig, axs = plt.subplots(2, 1, figsize=(11, 6))
for ax, m, tit, cmap, fmt in [(axs[0], cont, 'Média de acidentes por hora', 'Blues', '{:.0f}'),
                              (axs[1], let, 'Letalidade (mortos a cada 100 acidentes)', 'Oranges', '{:.0f}')]:
    im = ax.imshow(m.values, aspect='auto', cmap=cmap)
    ax.set_yticks(range(7), nomes_dias)
    ax.set_xticks(range(0, 24), [f'{h}h' for h in range(24)], fontsize=8)
    ax.grid(False)
    ax.set_title(tit)
    for (i, j), v in np.ndenumerate(m.values):
        ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=6.5,
                color='white' if v > np.nanpercentile(m.values, 70) else TINTA)
    fig.colorbar(im, ax=ax, fraction=.02, pad=.01)
salva(fig, 'g04_mapa_calor_dia_hora.png')

# G5 - volume x letalidade por hora (escalas separadas, sem eixo duplo)
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axs[0].bar(hora.index, hora['acidentes'] / 1000, color=AZUL, width=.75)
axs[0].set_title('Acidentes por hora do dia (milhares, 2020–2025)')
axs[1].bar(hora.index, hora['letalidade'], color=LARANJA, width=.75)
axs[1].set_title('Letalidade por hora do dia (mortos a cada 100 acidentes)')
axs[1].set_xticks(range(24), [f'{h}h' for h in range(24)], fontsize=8)
salva(fig, 'g05_volume_letalidade_hora.png')

# G6 - curva antes/durante/depois dos feriados
ref = dia['acidentes'].groupby(dia.index.dayofweek).mean()  # média por dia da semana (base de comparação)
x = df.dropna(subset=['dias_ate_feriado']).groupby(['dias_ate_feriado', 'data']).agg(a=('id', 'size'),
                                                                                    m=('mortos', 'sum')).reset_index()
x['esperado'] = x['data'].dt.dayofweek.map(ref)
x['indice'] = x['a'] / x['esperado'] * 100
x['letal'] = x['m'] / x['a'] * 100
curva = x.groupby('dias_ate_feriado').agg(indice=('indice', 'mean'), letalidade=('letal', 'mean'), dias=('data', 'size'))
curva.round(2).to_csv(os.path.join(TAB, 'curva_feriado.csv'), encoding='utf-8')
R['curva_feriado'] = curva.round(2).reset_index().to_dict('records')
fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(100, color=TINTA2, lw=1, ls='--')
ax.plot(curva.index, curva['indice'], color=AZUL, lw=2, marker='o', ms=8, mec='white', mew=2)
for i, v in curva['indice'].items():
    ax.text(i, v + 1.2, br(v, 0), ha='center', fontsize=9, color=TINTA)
ax.set_xticks(range(-3, 4), ['3 dias\nantes', '2 dias\nantes', 'véspera', 'feriado', 'dia\nseguinte', '2 dias\ndepois',
                             '3 dias\ndepois'])
ax.set_ylim(curva['indice'].min() - 3, curva['indice'].max() + 4)
ax.set(title='Acidentes em torno dos feriados (100 = média do mesmo dia da semana)', ylabel='Índice')
salva(fig, 'g06_curva_feriado.png')

# G7 - causas: volume x letalidade
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(causa['acidentes'] / 1000, causa['letalidade'], s=causa['mortos'] / 15, color=AZUL, alpha=.85,
           edgecolor='white', lw=2)
desloc = {'Clima e ambiente': (-4, -14), 'Animais e obstáculos': (-10, 9), 'Álcool e drogas (condutor)': (8, -3),
          'Via e sinalização': (8, 6), 'Falha do veículo': (8, 0), 'Distância de segurança': (8, -6),
          'Falta de atenção / reação do condutor': (-10, 12)}
for n, r in causa.iterrows():
    dx, dy = desloc.get(n, (8, 4))
    ax.annotate(n, (r['acidentes'] / 1000, r['letalidade']), xytext=(dx, dy), textcoords='offset points',
                fontsize=8, color=TINTA, ha='right' if dx < 0 else 'left')
ax.axhline(df['mortos'].sum() / len(df) * 100, color=TINTA2, ls='--', lw=1)
ax.text(ax.get_xlim()[0] + 1, df['mortos'].sum() / len(df) * 100 + .3, 'média geral', fontsize=8, color=TINTA2,
        ha='left', va='bottom')
ax.set(title='Grupos de causa: frequência x letalidade (tamanho = total de mortos)', xlabel='Acidentes (milhares)',
       ylabel='Mortos a cada 100 acidentes')
salva(fig, 'g07_causas_volume_letalidade.png')

# G8 - mortos por 100 mil habitantes, por UF
u = uf.sort_values('mortos_100mil_ano')
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(u.index, u['mortos_100mil_ano'], color=[LARANJA if i == 'BA' else AZUL for i in u.index], height=.7)
for i, v in enumerate(u['mortos_100mil_ano']):
    ax.text(v + .05, i, br(v, 1), va='center', fontsize=8, color=TINTA)
ax.set(title='Mortos em rodovias federais por 100 mil habitantes ao ano, por UF', xlabel='Mortos / 100 mil hab. / ano')
ax.grid(axis='y', visible=False)
salva(fig, 'g08_mortos_100mil_uf.png')

# G9 - chuva: letalidade por intensidade
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(clima.index.astype(str), clima['letalidade'], color=[CINZA, SEQ[2], SEQ[4], SEQ[6]], width=.6)
for i, (v, n) in enumerate(zip(clima['letalidade'], clima['acidentes'])):
    ax.text(i, v + .1, f'{br(v, 1)}\n(n={br(n)})', ha='center', fontsize=8, color=TINTA)
ax.set(title='Letalidade por intensidade da chuva na hora do acidente (INMET)', ylabel='Mortos / 100 acidentes')
ax.set_ylim(0, clima['letalidade'].max() * 1.3)
salva(fig, 'g09_chuva_letalidade.png')

# G10 - urbano x rural
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
rot = ['Rural', 'Urbano']
axs[0].bar(rot, urb['acidentes'] / 1000, color=AZUL, width=.55)
axs[0].set_title('Acidentes (milhares)')
axs[1].bar(rot, urb['letalidade'], color=LARANJA, width=.55)
axs[1].set_title('Mortos a cada 100 acidentes')
for ax, col in [(axs[0], urb['acidentes'] / 1000), (axs[1], urb['letalidade'])]:
    for i, v in enumerate(col):
        ax.text(i, v * 1.02, br(v, 1), ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, col.max() * 1.15)
salva(fig, 'g10_urbano_rural.png')

with open(os.path.join(OUT, 'resumo_eda.json'), 'w', encoding='utf-8') as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)
print(desc.round(2).to_string())
print(td.round(2).to_string())
print(ev.round(2).to_string())
print(curva.round(1).to_string())
print(causa.round(2).to_string())
print({k: v for k, v in R['trechos'].items() if k != 'top10'})
print(trecho.head(10).to_string())
print(urb.round(2).to_string()); print(fase.round(2).to_string()); print(pista.round(2).to_string())
print(ano.round(2).to_string())
print('dia recorde', R['dia_recorde'], R['dia_mais_mortes'])
