"""
ETL - Etapa 2: população por UF e ano (IBGE), 2020-2025.

Fontes:
  - Estimativas da População (IBGE), aba "BRASIL E UFs": 2020, 2021, 2024, 2025  -> IBGE/POPAAAA_*.xls
  - Censo Demográfico 2022 (IBGE), população por UF publicada no DOU em 2023 (POP_TCU_2023_Brasil_e_UFs_POP2022):
    https://ftp.ibge.gov.br/Informacoes_Gerais_e_Referencia/Relacao_da_Populacao_dos_Municipios_para_publicacao_no_DOU_em_2023/POP_TCU_2023_Brasil_e_UFs_POP2022_Malha2023.pdf
  - 2023: sem estimativa entre os arquivos obtidos -> média simples entre 2022 (Censo) e 2024 (estimativa).

Saída: saida/02_populacao_uf.csv  (uf, ano, populacao, fonte_populacao)
"""
import glob
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(BASE, 'saida')
os.makedirs(SAIDA, exist_ok=True)

UF = {'Rondônia': 'RO', 'Acre': 'AC', 'Amazonas': 'AM', 'Roraima': 'RR', 'Pará': 'PA', 'Amapá': 'AP',
      'Tocantins': 'TO', 'Maranhão': 'MA', 'Piauí': 'PI', 'Ceará': 'CE', 'Rio Grande do Norte': 'RN',
      'Paraíba': 'PB', 'Pernambuco': 'PE', 'Alagoas': 'AL', 'Sergipe': 'SE', 'Bahia': 'BA',
      'Minas Gerais': 'MG', 'Espírito Santo': 'ES', 'Rio de Janeiro': 'RJ', 'São Paulo': 'SP',
      'Paraná': 'PR', 'Santa Catarina': 'SC', 'Rio Grande do Sul': 'RS', 'Mato Grosso do Sul': 'MS',
      'Mato Grosso': 'MT', 'Goiás': 'GO', 'Distrito Federal': 'DF'}

# Censo 2022 - população residente por UF (IBGE, publicação no DOU em 2023). Total Brasil: 203.080.756
CENSO_2022 = {'RO': 1581196, 'AC': 830018, 'AM': 3941613, 'RR': 636707, 'PA': 8120131, 'AP': 733759,
              'TO': 1511460, 'MA': 6776699, 'PI': 3271199, 'CE': 8794957, 'RN': 3302729, 'PB': 3974687,
              'PE': 9058931, 'AL': 3127683, 'SE': 2210004, 'BA': 14141626, 'MG': 20539989, 'ES': 3833712,
              'RJ': 16055174, 'SP': 44411238, 'PR': 11444380, 'SC': 7610361, 'RS': 10882965, 'MS': 2757013,
              'MT': 3658649, 'GO': 7056495, 'DF': 2817381}
assert sum(CENSO_2022.values()) == 203_080_756


def numero(v):
    """'3.281.480(1)' -> 3281480 ; 211755692 -> 211755692"""
    if isinstance(v, (int, float)):
        return int(v)
    v = re.sub(r'\(\d+\)', '', str(v))
    return int(re.sub(r'\D', '', v))


linhas = []
for f in sorted(glob.glob(os.path.join(BASE, 'IBGE', 'POP*.xls'))):
    ano = int(re.search(r'POP(\d{4})', os.path.basename(f)).group(1))
    x = pd.read_excel(f, sheet_name='BRASIL E UFs', header=None)
    nome = x[0].astype(str).str.strip()
    # o número fica na primeira coluna numérica à direita do nome
    for i, n in nome.items():
        if n in UF:
            valor = next(v for v in x.iloc[i, 1:] if pd.notna(v) and re.search(r'\d', str(v)))
            linhas.append({'uf': UF[n], 'ano': ano, 'populacao': numero(valor),
                           'fonte_populacao': 'IBGE - Estimativa da População'})
pop = pd.DataFrame(linhas)
pop = pd.concat([pop, pd.DataFrame([{'uf': u, 'ano': 2022, 'populacao': p, 'fonte_populacao': 'IBGE - Censo 2022'}
                                    for u, p in CENSO_2022.items()])])
p22 = pop[pop.ano == 2022].set_index('uf')['populacao']
p24 = pop[pop.ano == 2024].set_index('uf')['populacao']
p23 = ((p22 + p24) / 2).round().astype(int)
pop = pd.concat([pop, pd.DataFrame({'uf': p23.index, 'ano': 2023, 'populacao': p23.values,
                                    'fonte_populacao': 'Média entre Censo 2022 e Estimativa 2024'})])
pop = pop.sort_values(['ano', 'uf']).reset_index(drop=True)

# verificações
cont = pop.groupby('ano')['uf'].nunique()
assert (cont == 27).all() and sorted(cont.index) == list(range(2020, 2026)), cont
print(pop.pivot(index='uf', columns='ano', values='populacao').to_string())
print('\nTotal Brasil por ano:')
print(pop.groupby('ano')['populacao'].sum().to_string())
pop.to_csv(os.path.join(SAIDA, '02_populacao_uf.csv'), index=False, encoding='utf-8')
