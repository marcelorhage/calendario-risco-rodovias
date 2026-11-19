"""
Etapa 7: dados do dashboard.

Gera um arquivo binário colunar compacto (1 linha por acidente, só os campos que o dashboard filtra e agrega),
para que todos os filtros funcionem no navegador sem servidor, e um JSON com dicionários e resultados estáticos.

Saída: dashboard/data/acidentes.bin   (colunas concatenadas, little-endian; layout descrito em meta.json)
       dashboard/data/meta.json       (dicionários, calendário, população, trechos, layout do binário)
       dashboard/data/estatico.json   (resultados do diagnóstico e da previsão)
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'dashboard', 'data')
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(os.path.join(BASE, 'saida', 'base_final.parquet'))
D0 = pd.Timestamp('2020-01-01')


def codifica(s, ordem=None):
    """categoria -> código inteiro (0..n-1); devolve códigos e lista de rótulos."""
    s = s.astype('object').where(s.notna(), 'Não informado').astype(str)
    rot = ordem if ordem else sorted(s.unique())
    faltam = [x for x in sorted(s.unique()) if x not in rot]
    rot = list(rot) + faltam
    mapa = {r: i for i, r in enumerate(rot)}
    return s.map(mapa).astype(np.uint8).values, rot


UFS = sorted(df['uf'].unique())
TIPO_DIA = ['Dia útil comum', 'Fim de semana comum', 'Ponto facultativo', 'Feriado', 'Feriadão', 'Período festivo']
EVENTOS = ['Nenhum', 'Carnaval', 'Semana Santa', 'Tiradentes', 'Dia do Trabalho', 'Corpus Christi', 'São João',
           'Independência', 'N. Sra. Aparecida', 'Finados', 'Proclamação da República', 'Consciência Negra',
           'Natal/Réveillon']
CHUVA = ['Sem chuva', 'Fraca', 'Moderada', 'Forte', 'Sem dado']
FASE = ['Amanhecer', 'Pleno dia', 'Anoitecer', 'Plena Noite']
PISTA = ['Simples', 'Dupla', 'Múltipla']

cols = {}
cols['d'] = ((df['data'] - D0).dt.days.values).astype(np.uint16)
cols['h'] = df['hora'].fillna(0).astype(np.uint8).values
cols['uf'], _ = codifica(df['uf'], UFS)
cols['urb'] = df['urbano'].astype(np.uint8).values
cols['td'], rot_td = codifica(df['tipo_dia'], TIPO_DIA)
cols['ev'], rot_ev = codifica(df['grupo_evento'], EVENTOS)
cols['jan'] = (df['dias_ate_feriado'].astype('float').fillna(4) + 3).astype(np.uint8).values  # 0..6; 7 = fora
cols['gc'], rot_gc = codifica(df['grupo_causa'])
cols['ta'], rot_ta = codifica(df['tipo_acidente_padronizado'])
cols['tp'], rot_tp = codifica(df['tipo_pista'], PISTA)
cols['fd'], rot_fd = codifica(df['fase_dia'], FASE)
cols['cm'], rot_cm = codifica(df['condicao_metereologica'])
ch = df['intensidade_chuva'].astype('object').where(df['tem_dado_clima'], 'Sem dado')
cols['ch'], rot_ch = codifica(ch, CHUVA)
cols['m'] = df['mortos'].clip(upper=255).astype(np.uint8).values
cols['fg'] = df['feridos_graves'].clip(upper=255).astype(np.uint8).values
cols['fe'] = df['feridos'].clip(upper=255).astype(np.uint8).values

# trechos: índice (0 = sem trecho) + centróide (mediana das coordenadas) + município mais frequente
tr = df.dropna(subset=['trecho']).groupby('trecho').agg(
    lat=('latitude', 'median'), lon=('longitude', 'median'), uf=('uf', 'first'), br=('br', 'first'),
    km=('km', 'min'), municipio=('municipio', lambda s: s.mode().iat[0]), n=('id', 'size'))
tr = tr.sort_values(['uf', 'br', 'km']).reset_index()
idx = {t: i + 1 for i, t in enumerate(tr['trecho'])}
cols['tr'] = df['trecho'].map(idx).fillna(0).astype(np.uint16).values

ordem = ['d', 'tr', 'h', 'uf', 'urb', 'td', 'ev', 'jan', 'gc', 'ta', 'tp', 'fd', 'cm', 'ch', 'm', 'fg', 'fe']
layout, off = [], 0
with open(os.path.join(OUT, 'acidentes.bin'), 'wb') as fh:
    for c in ordem:
        a = np.ascontiguousarray(cols[c])
        b = a.astype('<u2' if a.dtype == np.uint16 else 'u1').tobytes()
        layout.append({'col': c, 'tipo': 'u16' if a.dtype == np.uint16 else 'u8', 'offset': off})
        fh.write(b)
        off += len(b)

# calendário nacional (tipo de dia e evento de cada um dos 2.192 dias, sem o São João regional)
cal = pd.read_csv(os.path.join(BASE, 'Calendário de Feriados 2020–2026 - Feriados.csv'), encoding='utf-8-sig', dtype=str)
cal['data'] = pd.to_datetime(cal['data'], format='%d/%m/%Y')
cal = cal[cal['grupo_evento'] != 'São João']
dias = pd.DataFrame({'data': pd.date_range('2020-01-01', '2025-12-31')}).merge(cal, on='data', how='left')
MAP_T = {'Feriado nacional': 'Feriado', 'Ponto facultativo': 'Ponto facultativo',
         'Ponto facultativo (meio período)': 'Ponto facultativo', 'Período festivo': 'Período festivo', 'Feriadão': 'Feriadão'}
dias['td'] = dias['tipo'].map(MAP_T)
dias.loc[dias['td'].isna() & (dias['data'].dt.dayofweek >= 5), 'td'] = 'Fim de semana comum'
dias['td'] = dias['td'].fillna('Dia útil comum')
dias['ev'] = dias['grupo_evento'].fillna('Nenhum')

pop = pd.read_csv(os.path.join(BASE, 'saida', '02_populacao_uf.csv'))
meta = {
    'n': int(len(df)), 'inicio': '2020-01-01', 'layout': layout, 'bytes': off,
    'dic': {'uf': UFS, 'td': rot_td, 'ev': rot_ev, 'gc': rot_gc, 'ta': rot_ta, 'tp': rot_tp, 'fd': rot_fd,
            'cm': rot_cm, 'ch': rot_ch},
    'cal_td': [rot_td.index(x) for x in dias['td']], 'cal_ev': [rot_ev.index(x) for x in dias['ev']],
    'pop': {u: {str(a): int(p) for a, p in g[['ano', 'populacao']].values} for u, g in pop.groupby('uf')},
    'trechos': [[t, UFS.index(u), round(float(la), 4) if pd.notna(la) else None,
                 round(float(lo), 4) if pd.notna(lo) else None, str(mu)]
                for t, u, la, lo, mu in tr[['trecho', 'uf', 'lat', 'lon', 'municipio']].values],
}
with open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8') as fh:
    json.dump(meta, fh, ensure_ascii=False, separators=(',', ':'))

# resultados estáticos (diagnóstico e previsão)
res = json.load(open(os.path.join(BASE, 'saida', 'diagnostico', 'resultados.json'), encoding='utf-8'))
prev = pd.read_csv(os.path.join(BASE, 'saida', 'diagnostico', 'previsao_2026_diaria.csv'))
cl = df[df['tem_dado_clima']]
concord = pd.crosstab(cl['prf_registrou_chuva'], cl['choveu_3h'].astype(bool), normalize='index')
est = {
    'H': {k: res[k] for k in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']},
    'logistica': res['logistica'], 'carnaval': res['causalidade_carnaval'],
    'correlacao': res['correlacao_diaria_spearman'], 'correlacao_uf': res['correlacao_uf_spearman'],
    'validacao': res['preditiva_validacao_2025'],
    'previsao': {'data': prev['data'].tolist(), 'acidentes': prev['acidentes_previsto'].tolist(),
                 'acid_inf': prev['acidentes_ic95_inf'].tolist(), 'acid_sup': prev['acidentes_ic95_sup'].tolist(),
                 'mortos': prev['mortos_previsto'].tolist(), 'evento': prev['grupo_evento'].tolist(),
                 'tipo_dia': prev['tipo_dia'].tolist()},
    'previsao_feriados': res['previsao_2026']['feriados'],
    'previsao_total': {'acidentes': res['previsao_2026']['acidentes_ano'], 'mortos': res['previsao_2026']['mortos_ano']},
    'concordancia_prf_inmet_3h': {'prf_chuva_inmet_chuva': round(float(concord.loc[True, True]) * 100, 1),
                                  'prf_sem_chuva_inmet_chuva': round(float(concord.loc[False, True]) * 100, 1)},
    'cobertura_clima_pct': round(float(df['tem_dado_clima'].mean()) * 100, 1),
}
with open(os.path.join(OUT, 'estatico.json'), 'w', encoding='utf-8') as fh:
    json.dump(est, fh, ensure_ascii=False, separators=(',', ':'), default=str)

print(f'linhas={len(df):,} bytes={off:,} trechos={len(tr)}')
print({k: len(v) for k, v in meta['dic'].items()})
print(rot_gc); print(rot_ta); print(rot_cm)
print(os.path.getsize(os.path.join(OUT, 'meta.json')), os.path.getsize(os.path.join(OUT, 'estatico.json')))
