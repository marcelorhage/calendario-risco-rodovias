"""
ETL - Etapa 1: PRF (acidentes 2020-2025) + calendário de feriados.

Entrada : PRF/datatranAAAA/datatranAAAA.csv
          Calendário de Feriados 2020–2026 - Feriados.csv
Saída   : saida/01_prf_feriados.parquet
          saida/relatorio_etapa1.txt
"""
import glob
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(BASE, 'saida')
os.makedirs(SAIDA, exist_ok=True)
log = []


def L(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log.append(s)


# ============================================================ 1. EXTRAÇÃO
partes = []
for f in sorted(glob.glob(os.path.join(BASE, 'PRF', 'datatran*', '*.csv'))):
    d = pd.read_csv(f, sep=';', encoding='latin-1', dtype=str, keep_default_na=False)
    L(f'[extração] {os.path.basename(f)}: {len(d):,} linhas, {d.shape[1]} colunas')
    partes.append(d)
df = pd.concat(partes, ignore_index=True)
n0 = len(df)
L(f'[extração] total bruto: {n0:,}')

# ============================================================ 2. PADRONIZAÇÃO
NULOS = {'': np.nan, 'NA': np.nan, '(null)': np.nan, 'Não Informado': np.nan, 'Ignorado': np.nan}


def limpa(s):
    s = s.str.strip().str.replace(r'\s+', ' ', regex=True)
    return s.replace(NULOS)


def num(s):
    s = s.str.replace(',', '.', regex=False).replace({'': np.nan, 'NA': np.nan})
    return pd.to_numeric(s, errors='coerce')


texto = ['dia_semana', 'uf', 'municipio', 'causa_acidente', 'tipo_acidente', 'classificacao_acidente',
         'fase_dia', 'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo',
         'regional', 'delegacia', 'uop']
for c in texto:
    df[c] = limpa(df[c])

df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
df['data'] = pd.to_datetime(df['data_inversa'], format='%Y-%m-%d', errors='coerce')
df['br'] = pd.to_numeric(df['br'], errors='coerce').astype('Int64')
for c in ['km', 'latitude', 'longitude']:
    df[c] = num(df[c])
for c in ['pessoas', 'mortos', 'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos']:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
hm = pd.to_datetime(df['horario'], format='%H:%M:%S', errors='coerce')
df['hora'] = hm.dt.hour.astype('Int64')
df['horario'] = hm.dt.strftime('%H:%M')

# ============================================================ 3. QUALIDADE E INCONSISTÊNCIAS
L(f'[qualidade] ids duplicados: {df["id"].duplicated().sum()}')
df = df[~df['id'].duplicated()]
L(f'[qualidade] datas inválidas: {df["data"].isna().sum()}')
df = df[df['data'].notna() & df['data'].dt.year.between(2020, 2025)]
L(f'[qualidade] horários inválidos: {df["hora"].isna().sum()}')

# Coordenadas fora da caixa do território brasileiro são anuladas (não corrigidas por suposição)
fora = ~(df['latitude'].between(-34, 5.5) & df['longitude'].between(-74.5, -28.5))
L(f'[qualidade] coordenadas ausentes/fora do Brasil (anuladas): {fora.sum():,}')
df.loc[fora, ['latitude', 'longitude']] = np.nan
df['coord_valida'] = df['latitude'].notna()
df.loc[df['km'] < 0, 'km'] = np.nan

soma = df['mortos'] + df['feridos_leves'] + df['feridos_graves'] + df['ilesos'] + df['ignorados']
L(f'[qualidade] registros com soma de vítimas diferente de "pessoas": {(soma != df["pessoas"]).sum():,} (mantidos)')
L(f'[qualidade] classificação ausente: {df["classificacao_acidente"].isna().sum()} '
  '(reconstruída a partir de mortos/feridos)')
sem_class = df['classificacao_acidente'].isna()
df.loc[sem_class, 'classificacao_acidente'] = np.select(
    [df.loc[sem_class, 'mortos'] > 0, df.loc[sem_class, 'feridos'] > 0],
    ['Com Vítimas Fatais', 'Com Vítimas Feridas'], 'Sem Vítimas')

# ============================================================ 4. HARMONIZAÇÃO DE CATEGORIAS
# A PRF trocou a lista de causas em 2021: 2020 usa categorias antigas. Unificamos sinônimos e
# criamos 'grupo_causa', comparável entre todos os anos.
SIN_CAUSA = {
    'Transitar no Acostamento': 'Transitar no acostamento',
    'Ingestão de Álcool': 'Ingestão de álcool pelo condutor',
    'Ingestão de Substâncias Psicoativas': 'Ingestão de substâncias psicoativas pelo condutor',
    'Mal Súbito': 'Mal súbito do condutor',
    'Não guardar distância de segurança': 'Condutor deixou de manter distância do veículo da frente',
    'Ingestão de álcool ou de substâncias psicoativas pelo pedestre':
        'Ingestão de álcool e/ou substâncias psicoativas pelo pedestre',
    'Pedestre - Ingestão de álcool/ substâncias psicoativas':
        'Ingestão de álcool e/ou substâncias psicoativas pelo pedestre',
}
df['causa_padronizada'] = df['causa_acidente'].replace(SIN_CAUSA)

GRUPOS = {
    'Falta de atenção / reação do condutor': [
        'Falta de Atenção à Condução', 'Ausência de reação do condutor', 'Reação tardia ou ineficiente do condutor',
        'Acessar a via sem observar a presença dos outros veículos', 'Condutor usando celular', 'Frear bruscamente'],
    'Velocidade': ['Velocidade Incompatível', 'Participar de racha'],
    'Álcool e drogas (condutor)': ['Ingestão de álcool pelo condutor',
                                   'Ingestão de substâncias psicoativas pelo condutor'],
    'Manobras e desrespeito às regras': [
        'Ultrapassagem Indevida', 'Transitar na contramão', 'Manobra de mudança de faixa', 'Conversão proibida',
        'Retorno proibido', 'Trafegar com motocicleta (ou similar) entre as faixas', 'Transitar no acostamento',
        'Transitar na calçada', 'Estacionar ou parar em local proibido', 'Acesso irregular',
        'Desobediência às normas de trânsito pelo condutor', 'Desrespeitar a preferência no cruzamento',
        'Condutor desrespeitou a iluminação vermelha do semáforo',
        'Deixar de acionar o farol da motocicleta (ou similar)', 'Modificação proibida'],
    'Distância de segurança': ['Condutor deixou de manter distância do veículo da frente'],
    'Sono e condição de saúde': ['Condutor Dormindo', 'Mal súbito do condutor', 'Suicídio (presumido)',
                                 'Transtornos Mentais (exceto suicidio)'],
    'Pedestre': ['Entrada inopinada do pedestre', 'Pedestre andava na pista', 'Pedestre cruzava a pista fora da faixa',
                 'Falta de Atenção do Pedestre', 'Desobediência às normas de trânsito pelo pedestre',
                 'Ingestão de álcool e/ou substâncias psicoativas pelo pedestre'],
    'Falha do veículo': [
        'Defeito Mecânico no Veículo', 'Demais falhas mecânicas ou elétricas', 'Avarias e/ou desgaste excessivo no pneu',
        'Problema com o freio', 'Problema na suspensão', 'Carga excessiva e/ou mal acondicionada',
        'Deficiência ou não Acionamento do Sistema de Iluminação/Sinalização do Veículo', 'Faróis desregulados'],
    'Via e sinalização': [
        'Defeito na Via', 'Pista Escorregadia', 'Pista esburacada', 'Pista em desnível',
        'Afundamento ou ondulação no pavimento', 'Acostamento em desnível', 'Acumulo de água sobre o pavimento',
        'Acumulo de óleo sobre o pavimento', 'Acumulo de areia ou detritos sobre o pavimento', 'Ausência de sinalização',
        'Sinalização da via insuficiente ou inadequada', 'Sinalização encoberta', 'Sinalização mal posicionada',
        'Deficiência do Sistema de Iluminação/Sinalização', 'Iluminação deficiente', 'Curva acentuada',
        'Declive acentuado', 'Desvio temporário', 'Obras na pista', 'Falta de acostamento',
        'Falta de elemento de contenção que evite a saída do leito carroçável',
        'Faixas de trânsito com largura insuficiente', 'Redutor de velocidade em desacordo', 'Semáforo com defeito',
        'Sistema de drenagem ineficiente', 'Demais falhas na via', 'Restrição de Visibilidade',
        'Restrição de visibilidade em curvas horizontais', 'Restrição de visibilidade em curvas verticais',
        'Área urbana sem a presença de local apropriado para a travessia de pedestres'],
    'Clima e ambiente': ['Chuva', 'Neblina', 'Fumaça', 'Fenômenos da Natureza', 'Demais Fenômenos da natureza'],
    'Animais e obstáculos': ['Animais na Pista', 'Objeto estático sobre o leito carroçável', 'Obstrução na via',
                             'Obstrução Via tentativa Assalto', 'Agressão Externa'],
}
MAPA = {c: g for g, cs in GRUPOS.items() for c in cs}
df['grupo_causa'] = df['causa_padronizada'].map(MAPA)
L(f'[harmonização] causas sem grupo: {df.loc[df["grupo_causa"].isna(), "causa_padronizada"].value_counts().to_dict()}')
df['grupo_causa'] = df['grupo_causa'].fillna('Outros / não informado')

SIN_TIPO = {'Colisão com objeto estático': 'Colisão com objeto',
            'Colisão com objeto em movimento': 'Colisão com objeto',
            'Colisão lateral mesmo sentido': 'Colisão lateral',
            'Colisão lateral sentido oposto': 'Colisão lateral'}
df['tipo_acidente_padronizado'] = df['tipo_acidente'].replace(SIN_TIPO)

# ============================================================ 5. CALENDÁRIO DE FERIADOS
cal = pd.read_csv(os.path.join(BASE, 'Calendário de Feriados 2020–2026 - Feriados.csv'),
                  encoding='utf-8-sig', dtype=str)
cal['data'] = pd.to_datetime(cal['data'], format='%d/%m/%Y')
cal = cal[cal['data'].dt.year <= 2025]
NORDESTE = {'BA', 'SE', 'PE', 'AL', 'PB', 'RN', 'CE', 'PI', 'MA'}

# Datas nacionais valem para todas as UFs; o São João (feriado estadual) só para o Nordeste.
cal_nac = cal[cal['grupo_evento'] != 'São João'][['data', 'evento', 'tipo', 'grupo_evento']]
cal_sj = cal[cal['grupo_evento'] == 'São João'][['data', 'evento', 'tipo', 'grupo_evento']]
df = df.merge(cal_nac.rename(columns={'tipo': 'tipo_cal'}), on='data', how='left')
m_sj = df['uf'].isin(NORDESTE) & df['data'].isin(cal_sj['data']) & df['tipo_cal'].isna()
sj = df.loc[m_sj, ['data']].merge(cal_sj, on='data', how='left')
df.loc[m_sj, ['evento', 'tipo_cal', 'grupo_evento']] = sj[['evento', 'tipo', 'grupo_evento']].values

df['fim_de_semana'] = df['data'].dt.dayofweek >= 5
TIPO_DIA = {'Feriado nacional': 'Feriado', 'Feriado estadual': 'Feriado', 'Ponto facultativo': 'Ponto facultativo',
            'Ponto facultativo (meio período)': 'Ponto facultativo', 'Período festivo': 'Período festivo',
            'Feriadão': 'Feriadão'}
df['tipo_dia'] = df['tipo_cal'].map(TIPO_DIA)
df.loc[df['tipo_dia'].isna() & df['fim_de_semana'], 'tipo_dia'] = 'Fim de semana comum'
df['tipo_dia'] = df['tipo_dia'].fillna('Dia útil comum')
df['eh_data_especial'] = ~df['tipo_dia'].isin(['Fim de semana comum', 'Dia útil comum'])
df['grupo_evento'] = df['grupo_evento'].fillna('Nenhum')

# Distância em dias até o feriado/ponto facultativo mais próximo (−3 = 3 dias antes, +3 = 3 dias depois)
nucleo = np.sort(cal.loc[cal['tipo'].isin(['Feriado nacional', 'Ponto facultativo']), 'data'].unique())
datas = df['data'].values
pos = np.searchsorted(nucleo, datas)
ant = nucleo[np.clip(pos - 1, 0, len(nucleo) - 1)]
prox = nucleo[np.clip(pos, 0, len(nucleo) - 1)]
igual = nucleo[np.clip(pos, 0, len(nucleo) - 1)] == datas
d_ant = (datas - ant) / np.timedelta64(1, 'D')
d_prox = (prox - datas) / np.timedelta64(1, 'D')
dist = np.where(igual, 0, np.where(d_ant <= d_prox, d_ant, -d_prox))
df['dias_ate_feriado'] = pd.Series(np.where(np.abs(dist) <= 3, dist, np.nan)).round().astype('Int64').values

# ============================================================ 6. ENRIQUECIMENTO
df['ano'] = df['data'].dt.year
df['mes'] = df['data'].dt.month
df['dia_semana_num'] = df['data'].dt.dayofweek  # 0 = segunda-feira
df['faixa_horaria'] = pd.cut(df['hora'].astype(float), [-1, 5, 11, 17, 23],
                             labels=['Madrugada (00–05h)', 'Manhã (06–11h)', 'Tarde (12–17h)', 'Noite (18–23h)'])
df['madrugada_fds'] = df['fim_de_semana'] & (df['hora'] <= 5)
df['periodo_pandemia'] = df['data'].between('2020-03-11', '2021-12-31')
df['urbano'] = df['uso_solo'].eq('Sim')
df['acidente_fatal'] = df['mortos'] > 0
df['acidente_com_vitima'] = (df['mortos'] + df['feridos']) > 0
# Índice de gravidade = Unidade Padrão de Severidade (UPS, Denatran):
# 1 = só danos materiais; 4 = com feridos; 6 = com feridos em atropelamento; 13 = com mortos
df['indice_gravidade'] = np.select(
    [df['mortos'] > 0,
     (df['feridos'] > 0) & df['tipo_acidente'].eq('Atropelamento de Pedestre'),
     df['feridos'] > 0],
    [13, 6, 4], 1)
# Trecho de 10 km. A quilometragem das BRs recomeça em cada UF, por isso o trecho inclui a UF.
k0 = np.floor(df['km'] / 10) * 10
ok = df['br'].notna() & df['km'].notna()
df['trecho'] = None
df.loc[ok, 'trecho'] = ('BR-' + df.loc[ok, 'br'].astype(str).str.zfill(3) + '/' + df.loc[ok, 'uf'] + ' km '
                        + k0[ok].astype(int).astype(str) + '–' + (k0[ok] + 10).astype(int).astype(str))

# ============================================================ 7. CARGA
COLS = ['id', 'data', 'ano', 'mes', 'dia_semana', 'dia_semana_num', 'fim_de_semana', 'horario', 'hora',
        'faixa_horaria', 'madrugada_fds', 'tipo_dia', 'eh_data_especial', 'evento', 'grupo_evento',
        'dias_ate_feriado', 'periodo_pandemia', 'uf', 'municipio', 'br', 'km', 'trecho', 'uso_solo', 'urbano',
        'latitude', 'longitude', 'coord_valida', 'causa_acidente', 'causa_padronizada', 'grupo_causa',
        'tipo_acidente', 'tipo_acidente_padronizado', 'classificacao_acidente', 'fase_dia', 'sentido_via',
        'condicao_metereologica', 'tipo_pista', 'tracado_via', 'pessoas', 'mortos', 'feridos_leves',
        'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos', 'acidente_fatal', 'acidente_com_vitima',
        'indice_gravidade', 'regional', 'delegacia', 'uop']
df = df[COLS].sort_values(['data', 'horario']).reset_index(drop=True)
df.to_parquet(os.path.join(SAIDA, '01_prf_feriados.parquet'), index=False)

# ============================================================ 8. VERIFICAÇÕES DE INTEGRIDADE
L(f'[carga] registros finais: {len(df):,} de {n0:,} | colunas: {df.shape[1]}')
L('[verificação] acidentes e mortos por ano:')
L(df.groupby('ano').agg(acidentes=('id', 'size'), mortos=('mortos', 'sum')).to_string())
L('[verificação] tipo_dia:')
L(df['tipo_dia'].value_counts().to_string())
L('[verificação] grupo_causa:')
L(df['grupo_causa'].value_counts().to_string())
L('[verificação] dias_ate_feriado:')
L(df['dias_ate_feriado'].value_counts(dropna=False).sort_index().to_string())
L('[verificação] nulos por coluna (> 0):')
nz = df.isna().sum()
L(nz[nz > 0].to_string())
with open(os.path.join(SAIDA, 'relatorio_etapa1.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(log))
