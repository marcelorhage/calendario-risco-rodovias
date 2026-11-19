# Calendário de Risco nas Rodovias Federais (2020–2025)

Projeto A3 da UC **Análise de Dados e Big Data** (semestre 2026.2) — área temática **Mobilidade Urbana**.

**Pergunta central:** quando, onde e por que acontecem os acidentes de trânsito mais graves nas rodovias federais brasileiras, e como datas especiais, horários e condições climáticas alteram a frequência, a gravidade e as causas desses acidentes?

| Entrega | Onde está |
|---|---|
| Apresentação do projeto | este README |
| Relatório de ETL, EDA, prescrições e predições | [`relatorio/Relatório Final - Projeto A3.docx`](relatorio/) |
| Código-fonte do ETL e das análises | [`etl/`](etl/) |
| Dashboard interativo | **Link:** _(preencher após a publicação na Vercel)_ · código em [`dashboard/`](dashboard/) |
| Vídeo (storytelling) | **Link:** _(preencher após publicar no YouTube)_ |

## Principais resultados

- **406.255 acidentes e 33.963 mortes** nas rodovias federais de 2020 a 2025; alta de 12% desde 2021.
- **O feriado se comporta como um fim de semana.** Controlando dia da semana, mês e ano, a data especial não altera as mortes (+2,7%; p = 0,28). Fins de semana comuns têm 21,3 mortos por dia, contra 13,1 nos dias úteis. O risco extra está nos 3 dias anteriores ao feriado (+6% a +9% de acidentes).
- **A madrugada dobra a chance de morte** (razão de chances 2,1 em relação à tarde), em qualquer dia da semana.
- **A colisão frontal** é 6,8% dos acidentes e **31% das mortes**; a pista simples aumenta em 84% a chance de morte.
- **Pedestres** são 3% dos acidentes e 12% das mortes.
- **15,5% dos trechos de 10 km concentram metade das mortes.**
- **A chuva aumenta a frequência dos acidentes (+22%), mas não a letalidade.**
- A festa de rua do Carnaval não explica os acidentes (teste com os anos sem festa, 2021 e 2022).
- Previsão para 2026: 74.960 acidentes e 6.351 mortes (modelo binomial negativo, erro diário de 8% na validação de 2025).

## Fontes de dados (públicas)

| Fonte | Conteúdo | Acesso |
|---|---|---|
| Polícia Rodoviária Federal (PRF) | Acidentes agrupados por ocorrência, 2020–2025 | [Dados abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf) |
| Instituto Nacional de Meteorologia (INMET) | Medições horárias das estações automáticas, 2020–2025 | [Dados históricos](https://portal.inmet.gov.br/dadoshistoricos) |
| IBGE | Estimativas da População (2020, 2021, 2024, 2025) e Censo 2022 por UF | [Estimativas](https://www.ibge.gov.br/estatisticas/sociais/populacao/9103-estimativas-de-populacao.html) · [Censo 2022 por UF](https://ftp.ibge.gov.br/Informacoes_Gerais_e_Referencia/Relacao_da_Populacao_dos_Municipios_para_publicacao_no_DOU_em_2023/POP_TCU_2023_Brasil_e_UFs_POP2022_Malha2023.pdf) |
| Governo Federal | Calendário de feriados e pontos facultativos (portarias anuais do MGI) | `Calendário de Feriados 2020–2026 - Feriados.csv` |

## Estrutura do repositório

```
etl/                     scripts em Python, executados em ordem
  01_prf_feriados.py       limpeza da PRF, harmonização de causas, calendário e novas variáveis
  02_ibge_populacao.py     população por UF e ano (2023 = média entre 2022 e 2024)
  03_inmet_extracao.py     extração das medições horárias do INMET (um ano por execução)
  04_integracao.py         cruzamento com a estação mais próxima (até 50 km) e com a população
  05_eda.py                estatística descritiva, tabelas e gráficos
  06_diagnostico_predicao.py  testes de hipóteses, regressão logística, correlações, experimento natural e previsão
  07_dados_dashboard.py    dados compactos para o dashboard
saida/                   resultados (base final em Parquet, tabelas, gráficos, logs de verificação)
dashboard/               dashboard interativo (HTML + JavaScript, publicado na Vercel)
relatorio/               relatório final (.docx)
```

## Como reproduzir

1. Baixe os dados brutos nas fontes acima e organize-os assim (não estão no repositório pelo tamanho, cerca de 2,4 GB):
   - `PRF/datatranAAAA/datatranAAAA.csv` (2020 a 2025)
   - `INMET/AAAA/*.CSV` (arquivos das estações, descompactados)
   - `IBGE/POPAAAA_*.xls` (2020, 2021, 2024, 2025)
2. Instale as dependências: `pip install -r requirements.txt`
3. Rode, na ordem:
   ```
   python etl/01_prf_feriados.py
   python etl/02_ibge_populacao.py
   python etl/03_inmet_extracao.py 2020   # repita para 2021 a 2025
   python etl/04_integracao.py
   python etl/05_eda.py
   python etl/06_diagnostico_predicao.py
   python etl/07_dados_dashboard.py
   ```

A base final (`saida/base_final.parquet`) já está no repositório: 406.255 linhas e 63 colunas. A versão em CSV (222 MB) é gerada pelo script 04 e não é versionada por ultrapassar o limite do GitHub.

## Equipe

_(preencher com os integrantes)_
