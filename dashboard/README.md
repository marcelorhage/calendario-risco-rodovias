# Dashboard — Calendário de Risco nas Rodovias Federais

Página estática (HTML + JavaScript), sem servidor. Todos os filtros funcionam no navegador.

## Arquivos
- `index.html` — o dashboard (Chart.js 4.4.1 e Leaflet 1.9.4 via cdnjs; fontes Barlow via Google Fonts).
- `data/acidentes.bin` — 406.255 acidentes em formato colunar compacto (gerado por `etl/07_dados_dashboard.py`).
- `data/meta.json` — dicionários das categorias, calendário, população por UF e ano, trechos.
- `data/estatico.json` — resultados fixos do diagnóstico e da previsão.
- `vercel.json` — cache dos arquivos de dados.

## Testar no computador
Na pasta `dashboard`, rode `python -m http.server 8000` e abra http://localhost:8000
(abrir o `index.html` direto com duplo clique não funciona, porque o navegador bloqueia a leitura dos arquivos de dados).

## Publicar na Vercel
Opção 1 (pelo GitHub): suba o repositório, crie um projeto na Vercel a partir dele e defina
**Root Directory = `dashboard`** (Framework Preset: Other, sem comando de build).
Opção 2 (linha de comando): dentro da pasta `dashboard`, rode `npx vercel --prod`.

## Atualizar os dados
Rode os scripts de `etl/` na ordem (01 a 07). O script 07 regrava a pasta `dashboard/data`.
