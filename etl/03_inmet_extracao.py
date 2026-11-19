"""
ETL - Etapa 3: extração dos dados meteorológicos horários do INMET (estações automáticas), por ano.

Uso:   python etl/03_inmet_extracao.py 2020        (um ano por execução, por causa da memória)
Entrada: INMET/AAAA/**/*.CSV   (um arquivo por estação; 8 linhas de metadados + medições horárias em UTC)
Saída:   saida/inmet/inmet_AAAA.parquet   (estacao, data_hora_utc, chuva_mm, temperatura_c)
         saida/inmet/estacoes_AAAA.csv    (estacao, nome, uf, latitude, longitude, inicio, fim, horas_validas)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'saida', 'inmet')
os.makedirs(OUT, exist_ok=True)
ano = int(sys.argv[1])

arquivos = sorted(glob.glob(os.path.join(BASE, 'INMET', str(ano), '**', '*.CSV'), recursive=True)
               + glob.glob(os.path.join(BASE, 'INMET', str(ano), '**', '*.csv'), recursive=True))
print(f'{ano}: {len(arquivos)} arquivos')

medicoes, estacoes, falhas = [], [], []
for f in arquivos:
    try:
        meta = {}
        with open(f, encoding='latin-1') as fh:
            for _ in range(8):
                k, v = fh.readline().rstrip('\n').split(';')[:2]
                meta[k.strip().rstrip(':')] = v.strip()
        d = pd.read_csv(f, sep=';', encoding='latin-1', skiprows=8, usecols=[0, 1, 2, 7], header=0,
                        names=['data', 'hora', 'chuva_mm', 'temperatura_c'], decimal=',',
                        na_values=['-9999', '-9999,0', ''], dtype={'data': str, 'hora': str})
        # formatos de data/hora: '2024/01/01' + '0000 UTC'  (ou '2019-01-01' + '00:00')
        h = d['hora'].str.replace(' UTC', '', regex=False).str.replace(':', '', regex=False).str.zfill(4)
        dt = pd.to_datetime(d['data'].str.replace('-', '/', regex=False) + ' ' + h, format='%Y/%m/%d %H%M',
                            errors='coerce')
        d = pd.DataFrame({'estacao': meta['CODIGO (WMO)'], 'data_hora_utc': dt,
                          'chuva_mm': pd.to_numeric(d['chuva_mm'], errors='coerce').astype('float32'),
                          'temperatura_c': pd.to_numeric(d['temperatura_c'], errors='coerce').astype('float32')})
        d = d[d['data_hora_utc'].notna()]
        # valores fisicamente impossíveis -> ausentes
        d.loc[(d['chuva_mm'] < 0) | (d['chuva_mm'] > 150), 'chuva_mm'] = np.nan
        d.loc[(d['temperatura_c'] < -15) | (d['temperatura_c'] > 50), 'temperatura_c'] = np.nan
        d = d[d['chuva_mm'].notna() | d['temperatura_c'].notna()]
        medicoes.append(d)
        estacoes.append({'estacao': meta['CODIGO (WMO)'], 'nome': meta.get('ESTACAO'), 'uf': meta.get('UF'),
                         'latitude': float(meta['LATITUDE'].replace(',', '.')),
                         'longitude': float(meta['LONGITUDE'].replace(',', '.')),
                         'inicio': d['data_hora_utc'].min(), 'fim': d['data_hora_utc'].max(),
                         'horas_validas': int(d['chuva_mm'].notna().sum())})
    except Exception as e:  # arquivo corrompido ou fora do padrão: registra e segue
        falhas.append((os.path.basename(f), repr(e)[:120]))

m = pd.concat(medicoes, ignore_index=True)
antes = len(m)
m = m.drop_duplicates(['estacao', 'data_hora_utc'])
m['estacao'] = m['estacao'].astype('category')
m.to_parquet(os.path.join(OUT, f'inmet_{ano}.parquet'), index=False)

e = pd.DataFrame(estacoes)
# uma estação pode aparecer em mais de um arquivo no mesmo ano: consolida
e = (e.groupby('estacao').agg(nome=('nome', 'first'), uf=('uf', 'first'), latitude=('latitude', 'first'),
                              longitude=('longitude', 'first'), inicio=('inicio', 'min'), fim=('fim', 'max'),
                              horas_validas=('horas_validas', 'sum')).reset_index())
e.to_csv(os.path.join(OUT, f'estacoes_{ano}.csv'), index=False, encoding='utf-8')

print(f'{ano}: estações={len(e)} | medições={len(m):,} (duplicadas removidas: {antes - len(m):,}) '
      f'| chuva válida={m["chuva_mm"].notna().mean():.1%} | falhas={len(falhas)}')
for f, err in falhas:
    print('  falha:', f, err)
