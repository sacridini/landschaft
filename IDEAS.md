# Ideias para o landschaft (métricas de paisagem)

Ideias levantadas durante o desenho das métricas de paisagem (comparando com
landscapemetrics e pylandstats) que ficaram fora do escopo atual, para
implementar em outro momento. As fases combinadas estão no fim, para referência.

## Métricas e estatísticas extras

- Outras diferenças possíveis no `convention="fragstats"`: SHAPE e LSI na forma
  raster do FRAGSTATS (perímetro / perímetro mínimo, como o pylandstats) e o círculo
  circunscrito sobre todos os cantos das células de borda (o landscapemetrics usa só
  as células das linhas e colunas extremas). PROX com a área só da parte do vizinho
  dentro do raio (a documentação do FRAGSTATS é ambígua; usamos a área inteira).
  Sem o FRAGSTATS para comparar, as métricas FRAGSTATS-only foram testadas contra
  implementações diretas das definições.

## Uso e API

- **Gráficos** (matplotlib opcional): `show_patches`, `show_cores`, `show_lsm`,
  `show_correlation`.

## Exportação

- Tabelas de patch muito grandes (dezenas de milhões de linhas) poderiam ir direto
  para Parquet em pedaços (pyarrow), sem montar o DataFrame inteiro na memória.

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
- Fora da memória (`block_rows`): o ENN guarda todas as células de borda (~24 B cada);
  um ENN realmente fora da memória (p. ex. árvores por bloco em disco, ou busca por
  anéis de blocos vizinhos) tiraria esse limite. `labels()`/`to_raster()` por blocos
  poderiam gravar direto num GeoTIFF (reescrevendo os rótulos na passada de detalhe).
  Blocos processados em paralelo (hoje um bloco por vez, paralelo por faixas) ajudariam
  em rasters estreitos.
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
- Bit a bit: somas, médias e desvios reproduzem o `long double` de 80 bits do R
  (`src/cpp/lsm/extended.hpp`, emulado no MSVC) e `log`/`sqrt` usam o x87 como a
  libm do mingw (`rmath.hpp`, `x87_msvc.asm`). Faltam `gyrate` (o R soma coordenadas
  e distâncias célula a célula em ordem de linha, em 80 bits: exigiria percorrer as
  células de cada patch em sequência; fora da memória, mais 3 passadas no arquivo),
  `pafrac` (QR do `lm()`: portar `dqrdc2`/`dqrsl` e o `dnrm2` do BLAS de referência
  do R 4.4) e alguns `enn` com células quase quadradas (a busca podada do R).
  `circle` não tem como: o R embaralha os pontos ao acaso.
- As somas sequenciais em 80 bits custam: num mapa sal-e-pimenta com 10 M de
  patches, as estatísticas de paisagem (uma cadeia sequencial por métrica) levam
  ~0,7 s. Uma adição de `double` mais rápida (caminho rápido, 5 ns) já existe; a
  subtração/multiplicação genéricas (~9 ns, latência) poderiam ter caminhos rápidos
  sem desvios.

## Fases combinadas (referência)

1. Motor C++ + área/borda, forma, agregação, diversidade, complexidade. **Feita.**
2. Área core (`core`, `cai`, `ncore`, `tca`, `cpland`, `ndca`, `dcad`, `dcore_*`). **Feita.**
3. ENN, círculo circunscrito. **Feita** (PROX continua nas ideias acima).
4. Vários rasters em paralelo, zonas/buffers/amostragem (`sample_lsm`), janela
   móvel (`window_lsm`). **Feita.**
5. Processamento fora da memória por tiles, juntando patches cortados nas costuras.
   **Feita** (`block_rows`), com todas as métricas idênticas ao cálculo em memória.
   Proposta original (2026-09-26):
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
6. Estatísticas extras do FRAGSTATS (`_am`, `_md`, `_ra`, `perim_*`), nomes do
   FRAGSTATS/pylandstats, `n_threads` e valores bit a bit iguais ao R. **Feita.**
7. Utilitários do landscapemetrics (`get_centroids`, `get_nearestneighbour`,
   `get_circumscribingcircle`, `get_adjacencies`, `get_boundaries`, `get_patches`,
   `extract_lsm`, `check_landscape`, `calculate_correlation`), bit a bit. **Feita.**
   Fora da memória (`block_rows`) esses utilitários não existem (precisam do raster).
8. Métricas do FRAGSTATS que o landscapemetrics não tem (`prox`, `simi`, `connect`,
   `econ`, `cwed`, `teci` e as estatísticas de `prox`/`simi`/`econ`) e
   `convention="fragstats"` (perímetro físico com células não quadradas, `dcore_*`
   sobre os tamanhos das áreas core). **Feita.**
9. Exportação e integração: `write_table` (CSV, Parquet, GeoPackage...),
   `zonal_metrics` (GeoDataFrame por zona), `to_raster` de métricas de classe e com
   gravação, `scale_sample` (bit a bit com o landscapemetrics) e `batch()` com
   `"landscape_metrics"` / `"window_metrics"` (tabelas CSV ou Parquet). **Feita.**
