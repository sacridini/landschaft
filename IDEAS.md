# Ideias para o landschaft (métricas de paisagem)

Ideias levantadas durante o desenho das métricas de paisagem (comparando com
landscapemetrics e pylandstats) que ficaram fora do escopo atual, para
implementar em outro momento. As fases combinadas estão no fim, para referência.

## Métricas e estatísticas extras

- **`_am`, `_md`, `_ra`** (média ponderada pela área, mediana, amplitude), como no
  pylandstats e no FRAGSTATS. O landscapemetrics só tem `_mn`, `_sd` e `_cv`. As
  reduções são genéricas no C++ (`reduce()` em `src/cpp/lsm/metrics.cpp`), então
  bastaria somar as três ao catálogo, sem referência do R (conferir contra o
  pylandstats).
- **Métricas do FRAGSTATS que não estão no landscapemetrics**: PROX (índice de
  proximidade, com `search_radius`; o pylandstats tem), SIMI (similaridade),
  CONNECT (conectância), contraste de borda (CWED, TECI, ECON, com matriz de pesos),
  GYRATE_AM (correlation length), RMN/FRAGSTATS "related circumscribing circle" _am.
- **Modo "FRAGSTATS estrito"** (opção, p. ex. `convention="fragstats"`) que corrija
  convenções do landscapemetrics:
  - perímetro com células não quadradas: o landscapemetrics multiplica as arestas
    entre vizinhos esquerda/direita pela resolução x (fisicamente seria a y);
  - `dcore_*` resumem o *número* de áreas core por patch (`ncore`), e não o
    tamanho das áreas core disjuntas como no FRAGSTATS;
  - `calculate_lsm` usa `neighbourhood` também em `te`, `ed`, `iji` e `contag`
    (seguimos as funções `lsm_*` individuais, que usam sempre 4 vizinhos).

## Uso e API

- **Nomes completos como aliases** (`total_area`, `shannon_diversity_index`, ...)
  e abreviações FRAGSTATS em maiúsculas (`TA`, `SHDI`), para quem vem do pylandstats
  ou do FRAGSTATS.
- **`check_landscape()`**: relatório de validade como o do landscapemetrics (CRS
  projetado, valores inteiros, número de classes).
- **Utilitários do landscapemetrics**: `get_centroids`, `get_nearestneighbour`,
  `get_circumscribingcircle`, `get_adjacencies`, `get_boundaries`, `extract_lsm`
  (valores das métricas de patch em pontos).
- **Correlação entre métricas** (`calculate_correlation` / `show_correlation`).
- **Gráficos** (matplotlib opcional): `show_patches`, `show_cores`, `show_lsm`.
- **`scale_sample`**: métricas em várias escalas (tamanhos de buffer) ao redor de
  pontos.
- **Integração com `batch()`**: `"landscape_metrics"` em `BATCH_TOOLS`, gravando
  uma tabela por raster (CSV/Parquet) e o JSON de registro.
- **`n_threads`** por chamada (hoje o núcleo usa todas as threads do OpenMP).

## Exportação

- **Parquet** (pyarrow opcional) para tabelas de patch grandes (milhões de linhas).
- **GeoPackage / GeoDataFrame** (geopandas opcional) para resultados zonais:
  métricas por polígono já com a geometria.
- **Rasters de métricas**: `to_raster()` já existe para métricas de patch. Poderia
  aceitar métricas de classe (pintando cada classe) e gravar direto com
  `write_raster`.

## Análises no estilo pylandstats

- **Assinatura espacial** (`SpatialSignatureAnalysis`): PCA + agrupamento de
  paisagens/zonas pelas métricas, com gráficos dos clusters.
- **Séries temporais**: `SpatioTemporalAnalysis` com gráficos de métrica × tempo,
  e as variantes zonal/buffer/grade no tempo.

## Desempenho e escala

- Paralelizar a inicialização (mapa de classes, contagens por classe) e a junção
  das peças entre faixas, que ainda são sequenciais (~0,4 s + 0,3 s em 100 M pixels).
  Com muitos patches o rótulo pesa mais: 40 M de patches em 100 M pixels levam ~7 s
  (junção, ordenação e permutação das tabelas por patch).
- ENN: a construção das árvores e a passada estável por classe ainda são limitadas
  pela memória (~0,5 s cada em 100 M pixels); nós menores (sem `begin`/`end` nos
  nós internos, caixas em 16 bits relativas) reduziriam o tráfego.
- Rótulos em 64 bits para rasters em memória com mais de 2^31 pixels (hoje esses
  casos ficam para o processamento por tiles).
- Janela móvel com métricas de patch (reclassificar por janela é caro): reaproveitar
  rótulos globais e só corrigir patches cortados pela borda da janela. Hoje cada janela
  é uma paisagem pequena (1000×1000 com janela 21×21 e as 66 métricas: 9 s); o custo
  restante é montar a paisagem (alocações) e ENN/core. Reaproveitar os buffers do
  `Landscape` entre janelas cortaria boa parte.
- `sample_metrics` com círculos e polígonos: a máscara usa `rasterio.features.rasterize`
  (GDAL, idêntico ao terra), ~0,4 ms por plot só de overhead Python. Portar a
  rasterização "all touched" do GDAL para C++ (com as mesmas regras de borda) deixaria
  10 mil plots circulares tão rápidos quanto os quadrados (5,4 s → ~1,6 s).

## Observações para os testes

- O círculo circunscrito do landscapemetrics usa um algoritmo aleatorizado: regerar
  as referências muda `circle` em ~1e-16 (1 ulp). Comparar com tolerância relativa
  e não regravar as referências sem necessidade.
- O landscapemetrics não calcula `circle` com células não quadradas (erro); o script
  de referência pula essas métricas nesse caso (mesmo teste `all.equal` do pacote).
  O landschaft calcula com a largura e a altura reais da célula.
- Ler as referências sempre com `float_precision="round_trip"` (o parser rápido do
  pandas muda o último bit de alguns valores).
- `_sd`/`_cv` diferem do R no último bit: o R (mingw) acumula em `long double` de
  80 bits e o MSVC não tem esse tipo. Para bater bit a bit, acumular com soma
  compensada (double-double) nas reduções.

## Fases combinadas (referência)

1. Motor C++ + área/borda, forma, agregação, diversidade, complexidade. **Feita.**
2. Área core (`core`, `cai`, `ncore`, `tca`, `cpland`, `ndca`, `dcad`, `dcore_*`). **Feita.**
3. ENN, círculo circunscrito. **Feita** (PROX continua nas ideias acima).
4. Vários rasters em paralelo, zonas/buffers/amostragem (`sample_lsm`), janela
   móvel (`window_lsm`). **Feita.**
5. Processamento fora da memória por tiles, juntando patches cortados nas costuras.
   Proposta (2026-09-26, ainda não aprovada):
   - ler o raster em blocos de linhas (faixas), rotular cada bloco com o motor atual e
     juntar os pedaços de patch cortados nas costuras, como já se faz entre as faixas
     de 64 linhas em memória (union-find sobre os pedaços + soma das estatísticas);
   - primeira etapa: as métricas que dependem só das estatísticas somáveis por patch e
     das adjacências entre classes (área, borda, forma, agregação, diversidade,
     complexidade), exatas e idênticas ao landscapemetrics;
   - segunda etapa: ENN, círculo e core, que precisam de informação que cruza os
     blocos (células de borda de patches vizinhos, caixas envolventes, distância à
     borda com `edge_depth`);
   - API a definir (p. ex. `landscape_metrics(..., tiled=True)` ou um objeto próprio),
     no estilo do `mspa_tiled` já existente; testar comparando com o cálculo em memória
     e com o landscapemetrics em mapas pequenos cortados em blocos pequenos.
