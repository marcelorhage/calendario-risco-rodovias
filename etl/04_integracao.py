"""
ETL - Etapa 4: integração final (PRF + feriados) x INMET x IBGE  ->  base consolidada.

Cruzamento com o clima (por ano):
  1. Para cada acidente com coordenadas, busca as 5 estações do INMET mais próximas (distância de Haversine),
     ativas naquele ano, até 50 km.
  2. Converte a hora local do acidente para UTC pelo fuso da UF (UTC-3; UTC-4 em AM, RO, RR, MT, MS; UTC-5 no AC)
     e arredonda para cima até a hora cheia: o registro horário do INMET acumula a chuva da hora anterior
     (ex.: acidente às 14:35 local = 17:35 UTC -> registro das 18:00 UTC).
  3. Usa a estação mais próxima que tenha medição válida de chuva naquela hora.
  4. Soma a chuva das 3 últimas horas na mesma estação (chuva_3h_mm).

Entrada: saida/01_prf_feriados.parquet, saida/02_populacao_uf.csv, saida/inmet/*
Saída:   saida/base_final.parquet, saida/base_final.csv, saida/relatorio_etapa4.txt
"""
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(BASE, 'saida')
RAIO_KM, K, R_TERRA = 50, 5, 6371.0
FUSO = {'AC': 5, 'AM': 4, 'RO': 4, 'RR': 4, 'MT': 4, 'MS': 4}  # horas a somar para chegar ao UTC (padrão 3)
log = []


def L(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log.append(s)


df = pd.read_parquet(os.path.join(SAIDA, '01_prf_feriados.parquet'))
L(f'[entrada] acidentes: {len(df):,}')

# ------------------------------------------------------------------ hora UTC de referência
local = pd.to_datetime(df['data'].dt.strftime('%Y-%m-%d') + ' ' + df['horario'], format='%Y-%m-%d %H:%M')
utc = local + pd.to_timedelta(df['uf'].map(FUSO).fillna(3), unit='h')
df['_hora_utc'] = utc.dt.ceil('h')

partes = []
for ano in range(2020, 2026):
    a = df[df['ano'] == ano].copy()
    est = pd.read_csv(os.path.join(SAIDA, 'inmet', f'estacoes_{ano}.csv'))
    est = est[est['horas_validas'] > 0].reset_index(drop=True)
    med = pd.read_parquet(os.path.join(SAIDA, 'inmet', f'inmet_{ano}.parquet'))
    med['estacao'] = med['estacao'].astype(str)
    med =med.set_index(['estacao', 'data_hora_utc']).sort_index()

    arvore = BallTree(np.radians(est[['latitude', 'longitude']].values), metric='haversine')
    com_coord = a['coord_valida'].values
    dist, idx = arvore.query(np.radians(a.loc[com_coord, ['latitude', 'longitude']].values), k=K)
    dist_km = dist * R_TERRA

    a['estacao_inmet'] = None
    a['distancia_estacao_km'] = np.nan
    a['chuva_mm'] = np.nan
    a['temperatura_c'] = np.nan
    pos = np.where(com_coord)[0]
    for j in range(K):
        falta = a.iloc[pos]['chuva_mm'].isna().values & (dist_km[:, j] <= RAIO_KM)
        if not falta.any():
            break
        linhas = pos[falta]
        cod = est['estacao'].values[idx[falta, j]]
        chave = pd.MultiIndex.from_arrays([cod, a['_hora_utc'].values[linhas]])
        val = med.reindex(chave)
        ok = val['chuva_mm'].notna().values
        col = a.columns.get_loc
        a.iloc[linhas[ok], col('chuva_mm')] = val['chuva_mm'].values[ok]
        a.iloc[linhas[ok], col('temperatura_c')] = val['temperatura_c'].values[ok]
        a.iloc[linhas[ok], col('estacao_inmet')] = cod[ok]
        a.iloc[linhas[ok], col('distancia_estacao_km')] = dist_km[falta, j][ok]

    # chuva acumulada nas 3 últimas horas (hora do acidente + 2 anteriores), mesma estação
    tem = a['estacao_inmet'].notna()
    soma = np.zeros(tem.sum())
    for h in range(3):
        chave = pd.MultiIndex.from_arrays([a.loc[tem, 'estacao_inmet'].values,
                                           (a.loc[tem, '_hora_utc'] - pd.Timedelta(hours=h)).values])
        soma += med.reindex(chave)['chuva_mm'].fillna(0).values
    a.loc[tem, 'chuva_3h_mm'] = soma
    L(f'[clima] {ano}: estações ativas={len(est)} | acidentes com clima={tem.mean():.1%} '
      f'| distância mediana={a["distancia_estacao_km"].median():.1f} km')
    partes.append(a)

df = pd.concat(partes, ignore_index=True).drop(columns='_hora_utc')

# ------------------------------------------------------------------ variáveis de clima
df['tem_dado_clima'] = df['chuva_mm'].notna()
df['choveu'] = (df['chuva_mm'] > 0).astype('boolean')
df.loc[~df['tem_dado_clima'], 'choveu'] = pd.NA
df['choveu_3h'] = (df['chuva_3h_mm'] > 0).astype('boolean')
df.loc[~df['tem_dado_clima'], 'choveu_3h'] = pd.NA
# intensidade horária (critério da American Meteorological Society): fraca < 2,5; moderada 2,5–7,6; forte >= 7,6 mm/h
df['intensidade_chuva'] = pd.cut(df['chuva_mm'], [-0.01, 0, 2.5, 7.6, np.inf],
                                 labels=['Sem chuva', 'Fraca', 'Moderada', 'Forte'])
df['prf_registrou_chuva'] = df['condicao_metereologica'].isin(['Chuva', 'Garoa/Chuvisco', 'Granizo'])

# ------------------------------------------------------------------ população (IBGE)
pop = pd.read_csv(os.path.join(SAIDA, '02_populacao_uf.csv'))
df = df.merge(pop[['uf', 'ano', 'populacao']].rename(columns={'populacao': 'populacao_uf'}),
              on=['uf', 'ano'], how='left')
assert df['populacao_uf'].notna().all()

# ------------------------------------------------------------------ carga
df = df.sort_values(['data', 'horario']).reset_index(drop=True)
df.to_parquet(os.path.join(SAIDA, 'base_final.parquet'), index=False)
df.to_csv(os.path.join(SAIDA, 'base_final.csv'), index=False, encoding='utf-8')

# ------------------------------------------------------------------ verificações de integridade
L(f'[carga] registros: {len(df):,} | colunas: {df.shape[1]}')
L(f'[verificação] ids únicos: {df["id"].is_unique}')
L(f'[verificação] acidentes com dado de clima: {df["tem_dado_clima"].sum():,} ({df["tem_dado_clima"].mean():.1%})')
L(f'[verificação] distância à estação (km): '
  f'{df["distancia_estacao_km"].describe(percentiles=[.25, .5, .75]).round(1).to_dict()}')
L('[verificação] concordância PRF x INMET (linhas = PRF registrou chuva; colunas = INMET mediu chuva > 0):')
L(pd.crosstab(df['prf_registrou_chuva'], df['choveu'], normalize='index').round(3).to_string())
L('[verificação] idem, considerando chuva nas 3 últimas horas:')
L(pd.crosstab(df['prf_registrou_chuva'], df['choveu_3h'], normalize='index').round(3).to_string())
L('[verificação] por ano:')
L(df.groupby('ano').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum'),
                        com_clima=('tem_dado_clima', 'mean')).to_string())
with open(os.path.join(SAIDA, 'relatorio_etapa4.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(log))
