# Ideias para o landschaft (métricas de paisagem)

Ideias levantadas durante o desenho das métricas de paisagem (comparando com
landscapemetrics e pylandstats) que ficaram fora do escopo atual, para
implementar em outro momento. As fases combinadas estão no fim, para referência.

## Métricas e estatísticas extras

- `convention="fragstats"`: PROX com a área só da parte do vizinho dentro do raio (a
  documentação do FRAGSTATS é ambígua; usamos a área inteira). Sem o FRAGSTATS para
  comparar, as métricas FRAGSTATS-only foram testadas contra implementações diretas
  das definições.

## Uso e API

- Próxima parte (o que ainda falta da API do landscapemetrics): `full_name` (colunas
  `name`, `type` e `function_name` como no `calculate_lsm`), `return_raster` no
  `sample_metrics` (o recorte do raster de cada plot, como no `sample_lsm`) e os
  helpers públicos `construct_buffer`, `pad_raster`, `raster_to_points` e
  `get_unique_values`, conferidos contra o R como nas outras partes.

## Exportação


## Análises no estilo pylandstats


## Desempenho e escala

- ENN: a construção das árvores e a passada estável por classe ainda são limitadas
  pela memória (~0,5 s cada em 100 M pixels); nós menores (sem `begin`/`end` nos
  nós internos, caixas em 16 bits relativas) reduziriam o tráfego.
- Rótulos em 64 bits para rasters em memória com mais de 2^31 pixels (hoje esses
  casos ficam para o processamento por tiles).
- Janela móvel: reaproveitar rótulos globais e só corrigir os patches cortados pela
  borda da janela (algoritmo incremental). Hoje cada janela é uma paisagem pequena
  (1000×1000, janela 21×21, 66 métricas: ~7 s; ~73 µs por janela numa thread).
  Perfil (py-spy --native, build com /Zi): círculo, rótulo e core são as maiores
  partes, e as alocações ~10%: reaproveitar os buffers do `Landscape` entre janelas
  cortaria as alocações.
- Fora da memória (`block_rows`): o ENN guarda todas as células de borda (~24 B cada);
  um ENN realmente fora da memória (p. ex. árvores por bloco em disco, ou busca por
  anéis de blocos vizinhos) tiraria esse limite. `labels()`/`to_raster()` por blocos
  poderiam gravar direto num GeoTIFF (reescrevendo os rótulos na passada de detalhe).
  Blocos processados em paralelo (hoje um bloco por vez, paralelo por faixas) ajudariam
  em rasters estreitos.

## Conectividade em grafos (Conefor Sensinode / Makurhini)

Análise de 2026-09-26. O Conefor Sensinode 2.2 (`C:\Users\eduar\Documents\dev\Conefor
Sensinode 2.2`) é o motor original dos índices de Saura: C++ Borland com interface gráfica
só para Windows (cálculo em `landscape.cpp` e `sensinodeProgram.cpp`, em `float` de 32 bits).
Lê só tabelas (nós com atributo, conexões como distâncias, probabilidades ou adjacências)
e calcula NC, NL, H, CCP, LCP, IIC, F, AWF e PC, com `p = exp(ln(k)/d₀ · d)` e caminhos de
produto máximo (Dijkstra sobre `-log p`, poda por `minProb`), além de dI por remoção de nós
(frações intra, flux e connector), varPC (adição de nós) e nós de área zero. Não tem
betweenness (BC, BCIIC, BCPC), que só existe no Conefor 2.6 de linha de comando.
O Makurhini 3.0 (`C:\Users\eduar\Documents\dev\Makurhini`, R) cobre o fluxo inteiro:
- nós em vetor ou raster;
- distâncias (`distancefile`): centroide, borda, menor custo, commute-time;
- índices: `MK_dPCIIC` (frações intra/direct/step), `MK_dECA`, `MK_dPCIIC_links`,
  `MK_Focal_nodes` (IICf, PCf, CCI);
- ProtConn e frações (`MK_ProtConn`, `MK_ProtConnMult`, `MK_ProtConn_raster`);
- centralidades `MK_BCentrality` (via Conefor 2.6) e `MK_RMCentrality`;
- versões em grade (`MK_Connect_grid`, `MK_dECA_grid`) e o PC por SAMC (`MK_dPC_SAMC`);
- `MK_Fragmentation`, `dispersal_distance`, `probability_distance`, `test_metric_distance`.

No dI ele recalcula as distâncias do grafo inteiro a cada nó removido (lento com milhares
de nós). Todos são GPL, como o landschaft.

Já existe no landschaft: `gtn_components` e `gtn_importance` (ECA, DOC e dPC na versão
topológica do GTB: cores e bridges do MSPA, `p = 1`), `conefor_inputs`, `cost_distance` e
`least_cost_path`; no `lsm`, rotulagem, área, core, a árvore do ENN (base da distância borda
a borda), `grid_zones`, `zonal_metrics` e `SpatioTemporalAnalysis`.

Proposta: módulo novo com a arquitetura do `lsm` (motor C++/OpenMP em `src/cpp/conn/` e um
objeto Python que guarda os resultados intermediários):

```python
g = ls.HabitatGraph.from_raster(cover, habitat=[1], min_area=10, distance="edge", threshold=20_000)
g = ls.HabitatGraph.from_vector(gdf, attribute="area", distance="least-cost", resistance=res)
g = ls.HabitatGraph.from_conefor("nodes.txt", "distances.txt")
g.overall(["pc", "iic", "eca", "nc", "h", "awf"], distance=[1000, 5000], probability=0.5)
g.node_importance("pc", distance=5000, probability=0.5)  # dPC, intra, flux, connector
g.link_importance("pc", ...); g.centrality(["bc", "bcpc", "radial", "medial"], ...)
g.add_nodes(candidates)                                  # varPC (restauração)
ls.eca_change([g_2000, g_2020]); ls.protconn(pas, region, distance=10_000, probability=0.5)
ls.connectivity_grid(g, grid_zones(...))
```

- Motor em camadas:
  1. nós (rotulagem do `lsm`; vetores pela rasterização all-touched ou pela geometria);
  2. distâncias esparsas até o limiar (borda a borda pela árvore do ENN, menor custo por
     Dijkstra multi-fonte em paralelo por nó, commute-time por sistema linear esparso
     com Eigen);
  3. grafo em CSR, PC por Dijkstra de produto máximo por nó em paralelo, IIC por busca
     em largura, dI recalculando só os caminhos que passavam pelo nó removido e só no
     componente dele;
  4. ProtConn (frações sobre PC e ECA, região e transfronteira).
- `gtn_importance` vira um caso especial (grafo do MSPA com `p = 1`); `HabitatGraph.to_conefor`
  generaliza o `conefor_inputs`.
- Fases:
  1. motor de grafo (IIC, PC, ECA, NC, NL, H, CCP, LCP, F, AWF, dI com frações) lendo e
     escrevendo os arquivos do Conefor. **Feita** (0.30.0: `HabitatGraph`, conferido com
     o Makurhini a 1e-12 e com o Conefor 2.2 compilado, em precisão simples);
  2. nós de raster e vetor, distâncias euclidianas (centroide e borda) com limiar;
  3. distâncias de menor custo (reaproveitando o `cost_distance`);
  4. importância das ligações, varPC, dECA no tempo e índices em grade;
  5. ProtConn completo (uma ou várias regiões);
  6. centralidades (BC, BCIIC, BCPC, radial, medial) e focais (IICf, PCf, CCI);
  7. opcionais: commute-time e SAMC (Fletcher et al. 2023).
- Decidido: o Makurhini (instalado na biblioteca do RStudio) é a referência, e o
  Conefor 2.2 compilado (`tools/conefor`) é conferido na precisão simples dele.
- Desempenho do PC em grafos densos (milhares de nós com `pij_min` baixo): o Dijkstra
  de todas as fontes é limitado pela memória (5000 nós, 4,3 M pares: 14 s dos 19 s).
  Ideias: linhas ordenadas por custo com poda (parar a varredura quando
  `d_v + w` passa do maior custo provisório, depois que todos os nós foram
  alcançados); custos em `float` só para filtrar; quando as distâncias são métricas
  (centroides), o caminho direto já é o melhor para os pares ligados.
- A decidir:
  - vetores com geopandas/shapely como dependências opcionais (extra `vector`);
  - se ProtConn e SAMC entram no escopo.

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
  células de cada patch em sequência; fora da memória, mais 3 passadas no arquivo)
  e alguns `enn` com células quase quadradas (a busca podada do R). `pafrac` já é
  bit a bit (QR do `lm()` portado, `rlm.cpp`).
  `circle` não tem como: o R embaralha os pontos ao acaso.
- Plataformas: MSVC x64 usa a x87 via MASM (`x87_msvc.asm`); gcc/clang em x86-64
  (Linux, macOS Intel) usam `long double` nativo e asm inline para log/sqrt; MinGW no
  Windows emula (`SoftExtended`), pois o Windows deixa a x87 em 53 bits. Em ARM64
  (Apple M1–M6, Linux aarch64) as somas são emuladas (os mesmos bits do R em x86),
  mas `log`/`sqrt` vêm da libm do sistema: podem diferir no último bit do R-Windows
  nas métricas com logaritmo (shdi, ent, frac...). Falta: CI em Linux/macOS/ARM
  (cibuildwheel no GitHub Actions) com os testes bit a bit tolerando só essas, ou um
  `log` x87 emulado em software (`fyl2x`/`fyl2xp1` com 64 bits) para ser bit a bit
  também em ARM. OpenMP no macOS (Apple clang não traz; libomp) também é pendência.

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
10. Gráficos com matplotlib (opcional): `show_patches`, `show_cores`, `show_lsm`,
    `show_correlation`. **Feita.**
11. Análises no estilo pylandstats: `SpatioTemporalAnalysis` (séries de métricas no
    tempo, de mapas inteiros, zonas, buffers ou grade, com gráficos) e
    `SpatialSignatureAnalysis` (PCA e k-means das assinaturas, com gráficos). **Feita.**
12. Desempenho (parte): rótulo com somas, radix sort e permutações em paralelo;
    laços pequenos fora do OpenMP; rasterização "all touched" do GDAL portada para C++
    (plots circulares e polígonos: 10 mil círculos 5,4 s → 2,2 s, as mesmas células).
    **Feita.** O que falta está em "Desempenho e escala".
13. `pafrac` bit a bit: o `lm()` do R portado (`dqrdc2`/`dqrsl` do LINPACK sobre o BLAS
    de referência do R, com o `dnrm2` do LAPACK 3.12), e as janelas móveis
    transpostas como no `window_lsm` (ordem dos patches do R). **Feita.**
14. `convention="fragstats"` completa: SHAPE raster (perímetro / perímetro mínimo),
    LSI físico com células não quadradas, desvio padrão sobre n nos `_sd`/`_cv`, e o
    círculo circunscrito sobre o patch inteiro (fecho convexo exato das extremidades
    de cada linha, também fora da memória). **Feita.**
15. Tabelas grandes em pedaços: `Landscape.write_metrics` / `write_patch_metrics`
    (Parquet com um row group por pedaço, CSV, CSV.gz) e o `batch("landscape_metrics")`
    gravando assim. **Feita.**
16. `SpatialSignatureAnalysis`: NMF (NNDSVDa + HALS, como o scikit-learn), agrupamento
    hierárquico (ligações do scipy) e o clustergrama (`clustergram`,
    `plot_clustergram`, como o `get_cgram` do pylandstats). **Feita.**
17. Desempenho: inicialização paralela (faixa de valores, histograma por thread, sem
    zerar o raster de classes; 100 M pixels: 0,43 s → 0,07 s), ENN de mapas pequenos
    por pares de patches com SSE2, `stat_values` sem cópias, nomes comparados como
    `string_view`, laços `omp if` trocados por `parallel_for` (janelas: 130 → 97 µs
    por janela numa thread; 1000×1000 com 66 métricas: 12,0 s → 8,7 s). **Feita.**
18. Somas do R na x87 (MSVC x64: `sum`, `mean`, `var` e `sum_of` em MASM, os mesmos
    bits; estatísticas em 27 M de patches 5–8× mais rápidas, janelas 97 → 73 µs) e
    portabilidade: o build com gcc (MinGW) passa todos os testes (emulação no Windows,
    gyrate em `double` em todo lugar), MASM e intrínsecos só em x64. **Feita.**
