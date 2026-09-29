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

- O que faltava da API do landscapemetrics: `full_name`, `return_raster` no
  `sample_metrics` e os helpers `construct_buffer`, `pad_raster`, `unpad_raster`,
  `raster_to_points` e `get_unique_values`. **Feito** (0.49.0, bit a bit com o R; o
  `sin` do círculo é o `fsin` x87; o extent dos recortes é o do terra num raster intacto:
  o `crop(mask = TRUE)` do terra reescreve o extent do raster recortado, e no
  `sample_lsm` os plots depois do primeiro saem deslocados alguns ulps, o que não
  reproduzimos).

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
  2. nós de raster e vetor, distâncias euclidianas (centroide e borda) com limiar.
     **Feita** (0.31.0: `from_raster`, `from_vector`, `node_table`, `distance_table`, os
     mesmos bits do Makurhini; o `keep` do Makurhini, simplificação com `rmapshaper`,
     ficou de fora);
  3. distâncias de menor custo. **Feita** (0.32.0: o `costDistance` do gdistance bit a
     bit, sem o `bounding_circles` nem o caminho em Java do Makurhini);
  4. importância das ligações, varPC, dECA no tempo e índices em grade. Ligações,
     mudanças de ligação e nós a adicionar (restauração, modo "adding nodes" do Conefor):
     **feita** (0.33.0); dECA no tempo **feito** (0.34.0, `eca_change`); os índices em grade
     (`MK_Connect_grid`, `MK_dECA_grid`) dependem do ProtConn (recorte dos nós por célula,
     áreas transfronteiriças) e passam para a fase 5;
  5. ProtConn completo (uma ou várias regiões). **Feito** (0.35.0, `protconn`,
     `protconn_regions`; sem a simplificação do mapshaper, que só muda os nós quando
     altera as formas); ProtConn em raster e índices em grade **feitos** (0.36.0,
     `protconn_raster`, `make_grid`, `connectivity_grid`, `eca_change_grid`);
  6. centralidades (BC, BCIIC, BCPC, radial, medial) **feitas** (0.37.0,
     `node_importance(["BC", "BCIIC", "BCPC"])`, `centrality`; conferidas com o
     `MK_RMCentrality` e com a linha de comando do Conefor 2.7.1, sem o Louvain, que é
     aleatório); focais **feitos** (0.38.0, `focal_nodes` = `MK_Focal_nodes`, sem a
     simplificação do mapshaper nem as estatísticas de fragmentação);
  7. opcionais: commute-time **feito** (0.39.0, `distance="commute-time"`, o
     `commuteDistance` do gdistance a 1e-11); SAMC **feito** (0.40.0,
     `settlement_probabilities`, `HabitatGraph.from_samc` = `MK_dPC_SAMC`, a 1e-14 do
     samc; sem a amostragem aleatória de vértices nem o `keep` do mapshaper).
     Todas as fases do plano concluídas.
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

## Conectividade por teoria de circuitos (Omniscape / Circuitscape)

Análise de 2026-09-26. O Omniscape.jl 0.7 (`C:\Users\eduar\Documents\dev\Omniscape.jl`, Julia,
~1,7 mil linhas em `src/`) calcula conectividade omnidirecional sobre raster, sem manchas:
uma janela móvel circular (`radius`, mais `buffer`) centrada em cada alvo (blocos de
`block_size`², só onde a soma das fontes do bloco é > 0) injeta a força do bloco nas fontes
da janela (fora do bloco-alvo, proporcional à força de cada uma), aterra o centro e soma a
corrente dos nós num mapa acumulado (`cum_currmap`). Opcionais: `flow_potential` (a mesma
coisa com condutância 1), `normalized_cum_currmap` (a razão), correção de artefatos de bloco
(`calc_correction`, dois solves nulos), conectividade condicional (condições presentes e
futuras, `within` pela mediana, `equal` pela moda), reclassificação, `source_from_resistance`
com `r_cutoff`, `precision = single`. O núcleo numérico é o `compute_omniscape_current` do
Circuitscape.jl 6.1 (Laplaciano do grid, solve `cg+amg` ou `cholmod`, corrente nos nós).
Completa o módulo `conn` (commute-time e SAMC já são circuitos e cadeias de Markov) e não há
versão nativa em Python. Todos GPL/MIT, compatíveis.

Critério de igualdade (o bit a bit dos outros módulos não se aplica: o solver padrão é
iterativo com tolerância ~1e-6, a ordem dos alvos é sorteada com `randperm` e somada por
fatia de thread, e os testes do próprio Omniscape usam `rtol = 1e-6`):
- estrutura idêntica: mesmos alvos, máscara de NoData, células nulas e não nulas, bordas;
- contra `solver = cholmod`: diferença relativa ≤ ~1e-10;
- contra o padrão `cg+amg`: dentro da tolerância do solver (1e-6, como os testes deles);
- o nosso determinístico (mesmos bits com qualquer número de threads).

Circuitscape 6.1 (`compute_omniscape_current`, lido no fonte instalado): nós são as células com
condutância > 0 (os `-9999` do Omniscape caem fora), numeradas em ordem de coluna; arestas para
direita, baixo e as duas diagonais da direita com `(x+y)/2` e `(x+y)/(2√2)`; componentes conexas
do grafo inteiro, e só se resolve a que tem fonte e terra (a do centro); terra `Inf` = nó
removido, tensão 0; `cholmod` fatora com `shift = 10ε` e refina se o resíduo relativo passar
de 1e-5; `cg+amg` (agregação suavizada, Gauss-Seidel, `Pinv` no nível grosso, sem a
regularização do modo pairwise) para em `rtol = 1e-6` no resíduo precondicionado e repete com
`rtol/100` se o resíduo verdadeiro passar de 1e-4. Corrente do nó = `max(entrada, saída)` das
correntes de ramo `|G[i,j]| (v_i − v_j)`, ignorando ramos com `|b / max| < 1e-8` (máximo com
sinal separado para o triângulo superior e o inferior da matriz).

Peculiaridades a reproduzir (confirmar no fonte):
- `get_source` zera o bloco-alvo em `radius+buffer+1` do recorte, que perto das bordas não é
  o centro (o recorte foi cortado); `xupper_sub` compara com `ncols` do mapa inteiro;
- `clip` põe `missing` onde a distância ao centro é > raio (fontes no raio, condutância no
  raio + buffer); o `buffer` só acrescenta colunas/linhas de NoData às fontes;
- `calc_correction`: limiar `1e-6 * maximum(block_null_current)`, sensível a células perto
  dele (um solver diferente pode jogá-las do outro lado); zeros viram fator 1;
- `mode` do StatsBase desempata pela ordem de um `Dict`; `median` com `skipmissing`;
- `flow_potential` usa o retângulo inteiro com condutância 1 (sem o recorte circular);
- no Circuitscape: vizinhança 8 (ou 4 com `connect_four_neighbors_only`), média das
  condutâncias, diagonais /√2, terra `Inf` = nó removido (tensão 0), componentes sem terra,
  corrente do nó (convenção exata a ler no fonte do Circuitscape.jl, a clonar);
- `precision = single` faz tudo em `Float32`.

Proposta: motor C++ novo `src/cpp/circuit/` (`circuitcore`: `grid.cpp` Laplaciano como o
Circuitscape, `solve.cpp`, `omniscape.cpp`), Python `ls.omniscape(resistance, radius=...,
block_size=..., source=..., ...)` devolvendo um `xarray.Dataset` (`cum_currmap`,
`flow_potential`, `normalized`) e `ls.omniscape_ini(path)` para os `.ini` existentes. O Eigen
entra no projeto aqui (ainda não é usado em `src/cpp`; só cabeçalhos, FetchContent ou
vendorizado).

Desempenho (o Omniscape, por janela, recorta arrays com `missing`, monta o grafo, acha
componentes, monta o AMG do zero e roda o CG, duas vezes com flow potential):
- janelas interiores têm o mesmo formato (disco de `radius + buffer`): ordenação por nested
  dissection geométrica e fatoração simbólica uma vez; por janela só a numérica (NoData como
  condutância 0 e linha identidade no nó isolado, mantendo o padrão fixo);
- o Laplaciano nulo (condutância 1) é o mesmo em todas as janelas interiores: fatorar uma
  vez por formato (cache para as bordas) e o flow potential vira duas substituições
  triangulares;
- correção de artefatos calculada uma vez, como no original;
- OpenMP nos alvos (`schedule(dynamic)`), área de trabalho por thread, zero alocação no laço;
  acumulação em ordem fixa por blocos com coloração 2D (um acumulador do tamanho do mapa por
  thread, como no Omniscape, daria ~13 GB em 10k × 10k com 16 threads);
- risco: Cholesky custa ~n^1,5 e o AMG ~n; para raio grande (≥ ~300 px, ~280 mil nós) o AMG
  pode ganhar, e aí seria preciso um CG com AMG por agregação suavizada em C++. Medir antes.

Fases:
0. benchmark rápido: tempo por janela do Omniscape (Julia, várias threads) contra Eigen LDLT
   com nested dissection, CG+IC e CHOLMOD, com r = 25, 50, 100, 200; decide o solver.
   **Feita** (`tools/omniscape_window_benchmark.jl`; uma thread, disco de raio r, condutância
   aleatória, terra no centro, ms por janela):

   | r (nós) | Circuitscape cg+amg | Circuitscape cholmod | Eigen LDLT, ND fixa (monta + fatora) | só a substituição |
   |---|---|---|---|---|
   | 25 (2 mil) | 6,1 | 4,0 | 1,0 | 0,07 |
   | 50 (8 mil) | 20,8 | 15,6 | 4,9 | 0,25 |
   | 100 (31 mil) | 119 | 67 | 35 | 1,6 |
   | 200 (126 mil) | 510 | 268 | 234 | 8,9 |

   Eigen com AMD em vez da ND geométrica: 66 ms (r = 100) e 700 ms (r = 200); CG+IC e CG+Jacobi
   (1e-6) 3–10× mais lentos que o direto; SparseLU 2× mais lento. Decidido: Eigen
   `SimplicialLDLT` com ordenação por nested dissection geométrica, análise simbólica uma vez
   por formato de janela e só a fatoração numérica por alvo; o flow potential sai da fatoração
   nula em cache (só a substituição). Com flow potential o Omniscape paga dois solves: ~11× (r =
   25) a ~4× (r = 200) por janela. Para r ≥ ~150 um Cholesky supernodal/multifrontal (blocos
   densos nos separadores) é o próximo ganho, na fase 3. O cg+amg do Circuitscape refaz o solve
   com `rtol/100` quando o resíduo verdadeiro passa de 1e-4 (aconteceu em r = 100 e 200);
1. núcleo de circuito (Laplaciano, solve, corrente nos nós), conferido com o
   `compute_omniscape_current` com `cholmod`. **Feita** (0.41.0: `ls.circuit.advanced`,
   `src/cpp/circuit/` com `CellSystem`, o padrão fixo reaproveitável pelas janelas; a 1,2e-13
   do CHOLMOD, mesmas células com corrente; o cg+amg fica a 2e-7);
2. Omniscape básico (alvos, fontes, terra, correção, flow potential, normalizada,
   `mask_nodata`, com as peculiaridades), conferido com `test/output_verify` (test1–5) e
   referências novas. **Feita** (0.42.0: `ls.omniscape`, também `source_threshold`,
   `r_cutoff`, condutância e 4 vizinhos; conferido com o `run_omniscape` por INI nas
   configurações 1–3 do Omniscape e em mapas sintéticos, a 1,7e-11 do CHOLMOD. Janelas
   agrupadas por formato, um `CellSystem` por formato e thread, flow potential com a
   fatoração em cache; soma por linhas em ordem fixa. 300×300, r = 25, bloco 3, com flow
   potential: Omniscape 17,4 s com 20 threads, landschaft 0,87 s (8,2 s numa thread));
3. desempenho (reuso simbólico, cache do nulo, OpenMP, acumulação determinística) e
   benchmark final contra o Omniscape. **Feita** (0.43.0): Cholesky multifrontal próprio
   sobre a ND geométrica (`system.cpp`: fronts densos nos separadores, LLT/TRSM/SYRK do
   Eigen; r = 100: 26 → 12 ms, r = 200: 206 → 78 ms), estrutura simbólica por formato
   compartilhada entre threads, substituição direta fundida na fatoração, flow potential
   em lotes de até 8 alvos (`solve_many`), módulo `_circuit_avx2` (x86-64, carregado por
   `cpu_supports_avx2_fma`; buffers alinhados a 64 bytes para o resultado não depender do
   endereço). Com 20 threads: 800² r = 10 0,64 s (Omniscape 29,4 s), 700² r = 25 1,36 s
   (46,7 s), 1000² r = 100 12,4 s (275 s). Ficou para depois: com r grande e muitas threads
   a banda de memória limita (6× de 1 para 20 threads em r = 100; o fator de uma janela tem
   13 MB); folhas da ND de 4 a 64 células dão o mesmo tempo; o Omniscape com `cholmod` é
   tão lento quanto com `cg+amg`;
4. conectividade condicional, reclassificação, `source_from_resistance`, `r_cutoff`, `.ini`,
   `precision = single`. **Feita** (0.44.0: condições no C++ depois da escala das fontes,
   mediana `a/2 + b/2`, moda do StatsBase = o primeiro valor a passar a maior contagem em
   ordem de coluna; `ls.circuit.omniscape_ini` com o parser do Omniscape; conferido com os
   INI de teste do próprio Omniscape (1–7, 4a–4c, 32 bit, reclass) a 5e-13 do CHOLMOD; o
   `config7` tem resistência 0 e o Omniscape recusa, como nós; em `single` a aritmética
   fica em double: o Omniscape em float32 fica a até 2,4e-4);
5. opcionais: solver iterativo para raio grande; modos raster do Circuitscape (pairwise,
   one-to-all, all-to-one, advanced) sobre o mesmo núcleo. Decidido (2026-09-26): o solver
   iterativo não compensa (o direto ganha até em r = 200; o limite real é a memória); a
   fase vira o port dos modos raster do Circuitscape 6.1, conferido com a suíte de
   verificação dele (`test/output_verify`: `sg` pairwise, `mg` advanced, `oneToAll`,
   `allToOne`, com polígonos, máscara, pares incluídos, fontes variáveis, média de
   resistências, 4/8 vizinhos, mapas de tensão/corrente/acumulada/máxima e a matriz de
   resistências) e com o CHOLMOD dele:
   - 5a. Omniscape com janelas enormes: threads limitadas pela memória do fator por thread.
     **Feita** (0.45.0: `memory_limit`, metade da RAM por padrão, estimado pela estrutura
     simbólica da janela cheia; `attrs["threads"]`);
   - 5b. núcleo raster do Circuitscape e modo advanced completo (mapa de habitat como
     resistência ou condutância, `connect_using_avg_resistances`, polígonos de curto-circuito,
     máscara, fontes e terras de arquivo, as quatro políticas de conflito, terra como
     resistência, terras diretas, mapas de tensão e corrente, `set_null_*_to_nodata`).
     **Feita** (0.46.0: `circuitscape.cpp` = mapa de nós e grafo do Circuitscape, solve por
     componente com `frontal_analyze_graph` (nós de polígono na raiz); leitor ASCII em
     double (o GDAL daria float32); `circuitscape_advanced`, `circuitscape_ini` com
     `scenario = advanced`; referências: as entradas de teste do Circuitscape copiadas e
     rodadas com CHOLMOD (`tools/make_reference_circuitscape.py`, todos os modos já
     gravados), mg1–7 e ex_advanced a 5e-12; no modo advanced o Circuitscape ignora
     `set_null_*_to_nodata` e escreve 0 fora dos nós);
   - 5c. pairwise (pontos e regiões focais, pares incluídos/excluídos, matriz de
     resistências `.out`, corrente acumulada e máxima, mapas por par); uma fatoração por
     componente e muitos lados direitos (`solve_many`). **Feita** (0.47.0:
     `raster_pairwise` com um nó aterrado por componente, lotes de 8 colunas por thread,
     `SolveScratch` por thread sobre o mesmo fator; o atalho do Circuitscape reproduzido
     com a peculiaridade da linha 1 da matriz de tensões; regiões focais com um grafo por
     par (`region_polymap` = `create_new_polymap`), pares em paralelo; sg1–17 batem,
     sgVerify17 a 9e-10 relativo porque o CHOLMOD resolve a laplaciana singular com
     deslocamento 10ε);
   - 5d. one-to-all e all-to-one (fontes variáveis, `set_focal_node_currents_to_zero`).
     **Feita** (0.48.0: o driver em Python monta fontes e terras por ponto como o
     `run_onetoall`, e `circuitscape_multi_advanced` resolve os pontos em paralelo sobre um
     grafo, só no componente do ponto; os 25 casos a 5e-14);
   - 5e. `ls.circuit.circuitscape_ini(path)` e a gravação das saídas como o Circuitscape.
     **Feita** junto com 5b–5d (todos os cenários raster; mapas `.asc`/`.tif`,
     `_resistances.out`, `_resistances_3columns.out`, a configuração em `output_file`).
     Fase 5 concluída; o modo de rede (grafos) do Circuitscape continua fora.

Próximos passos sugeridos (2026-09-26, depois da 0.48.0):
- Fatoração paralela de um componente grande. **Feita** (0.51.0: partes da árvore em
  paralelo, os fronts acima por alturas, fronts grandes por Cholesky em blocos fixos;
  mesmos bits com qualquer número de threads; 1000×1000: fatoração 3,4 → 0,7 s com 20
  threads). O que sobra no advanced de um componente só é serial: montar o grafo, o
  `factor_component` e as correntes (~1,6 s dos 2,6 s num raster 1000×1000), e no
  pairwise as substituições (lotes de 8 colunas relendo o L inteiro). Plano original:
  - paralelismo na árvore: subárvores independentes da ND fatoradas ao mesmo tempo, por
    alturas (OpenMP 2.0 do MSVC não tem `task`: `parallel for` sobre os nós de mesma
    altura), com as matrizes de atualização em áreas próprias por nó em vez da pilha única
    (o pico de memória sobe; somar os `u²` por nível para estimar);
  - paralelismo nos kernels densos dos fronts do topo (os separadores de ~N células
    concentram a maior parte dos flops: só a árvore dá no máximo ~3×): LLT/TRSM/SYRK em
    blocos com `parallel for`, ou o GEMM paralelo do Eigen onde couber;
  - manter o resultado determinístico (mesma ordem de soma com qualquer número de
    threads) e conferir com o CHOLMOD como nas outras partes.
- Modo de rede (grafos) do Circuitscape (`data_type = network`). **Feito** (0.50.0:
  `circuitscape_network_pairwise` / `_advanced`, `circuitscape_ini`; dissecção aninhada
  por níveis de BFS para grafos sem geometria, em paralelo por profundidade; os casos de
  rede do Circuitscape e seis sintéticos a 1,1e-14 do CHOLMOD; 1 M de nós, 20 pontos:
  7,2 s contra 27,7 s do CHOLMOD e 182 s do cg+amg; 5,0 s com a fatoração paralela).
- Omniscape com raio grande e muitas threads: limitado pela banda de memória (6× de 1 para
  20 threads em r = 100; o fator de uma janela tem 13 MB). Ideias: fatorar o landscape em
  precisão mista não serve (condicionamento); reduzir o tráfego guardando só a parte de L
  que a substituição reversa lê, ou resolver lotes de alvos vizinhos na mesma thread para
  reaproveitar a cache.

Referência: Julia 1.13.0 (`C:\Users\eduar\AppData\Local\Programs\Julia-1.13.0`) com
Omniscape 0.7 (o clone local, `Pkg.develop`) e Circuitscape 6.1 num depot separado
(`C:\Users\eduar\AppData\Local\landschaft-julia`, `JULIA_DEPOT_PATH=...\depot`,
`--project=...\env`; `jl.sh` lá dentro chama o Julia assim); script
`tools/make_reference_omniscape.jl` gravando em `tests/data/reference/circuit`.

## Modelos neutros de paisagem (NLMR)

Análise de 2026-09-28. O NLMR 1.4.0 (`C:\Users\eduar\Documents\dev\NLMR`, R, GPL-3) simula
paisagens neutras: gradientes (`nlm_planargradient`, `nlm_edgegradient`,
`nlm_distancegradient`), aleatórios (`nlm_random`, `nlm_percolation`), agrupados
(`nlm_randomcluster`, `nlm_curds`, `nlm_neigh`, `nlm_randomrectangularcluster`), fractais
(`nlm_mpd`, `nlm_fbm`), campos gaussianos (`nlm_gaussianfield`), mosaicos
(`nlm_mosaicfield`, `nlm_mosaictess`, `nlm_mosaicgibbs`) e a classificação
`landscape_classify` (proporções ou quebras de Jenks). Não há nada disso no landschaft.

A aleatoriedade vem de dois lugares, e os dois precisam ser reproduzidos para o mesmo
`user_seed` dar o mesmo mapa:
- R (`set.seed`, `runif`, `rnorm`, `sample`, `sample.int`): Mersenne-Twister do R com a
  mistura inicial do `set.seed`, `unif_rand` com `fixup`, `norm_rand` por inversão
  (`qnorm5` do AS241 sobre `u = (int)(2^27 u1) + u2`, com o `log` x87 do mingw),
  `R_unif_index` por rejeição (`rbits` em pedaços de 16 bits), `sample` sem reposição
  (Fisher-Yates parcial), com `prob` (`revsort` + busca linear; Walker com mais de 200
  probabilidades relevantes; sem reposição com remoção), `sample2` com hash para
  `n > 1e7`; mais tarde `rpois`/`exp_rand` para o spatstat;
- C++ do NLMR (`mpd`, `neigh`, `randomrectangularcluster`): `std::mt19937` com as
  distribuições da libstdc++ do Rtools (GCC 13.3): `generate_canonical` com duas palavras de
  32 bits, normal polar com o valor guardado, inteiros pelo método de Lemire (64 bits) e o
  `std::shuffle` que sorteia duas posições de uma vez quando `n² < 2^32`.

Critério de igualdade: bit a bit com o NLMR para a mesma semente em tudo, menos nas
tesselações de Voronoi (`mosaictess`, `mosaicgibbs`): o `st_union` ordena os germes e o
`st_voronoi` do GEOS devolve as células na ordem da sua estrutura interna, e os valores são
atribuídos nessa ordem. Ali: germes bit a bit, a mesma partição (a menos de empates exatos)
e o mesmo conjunto de valores; a ordem do GEOS pode ser portada depois.

Peculiaridades a reproduzir (com `nlmr_compat=True`, o padrão; `False` faz o que a
documentação promete):
- `mpd`: quadrado de lado `max(nrow, ncol) + 1` (ímpar), menos uma célula de cada borda,
  então a saída não tem as dimensões pedidas; uma `normal_distribution` nova por célula
  (o segundo valor do método polar é jogado fora);
- `neigh`: `p_neigh` e `p_empty` passam por `float`; embaralhamento da libstdc++;
- `distancegradient`: `origin` indexa `[linhas, colunas]` no terra, apesar das verificações
  contra `ncol`;
- `randomcluster`: o preenchimento olha o mapa antes de preencher (sem propagação); empates
  por `sample(fili, 1)`, sem vizinho por `sample(prob = ai)` sem reposição;
- `randomrectangularcluster`: varre a matriz inteira depois de cada retângulo (O(N·k));
  um contador de células vazias dá o mesmo resultado;
- `fbm` e `gaussianfield` estão desativadas no NLMR 1.4.0 (só uma mensagem); a referência
  são as funções internas novas `NLMR:::simulate_fbm` e `simulate_gaussian_field`
  (espectral e embutimento circulante com o `fft` do R).

Proposta: motor C++ `src/cpp/nlm/` (`rng.hpp/.cpp` com os dois geradores, o `fft` do R
portado, um arquivo por família de modelos) e o subpacote Python `landschaft.nlm`
(`ls.nlm.mpd()`, `ls.nlm.random_cluster()`, ..., `ls.nlm.classify()`, `ls.nlm.rescale()`),
devolvendo `xarray.DataArray` com a resolução, `seed` (o `user_seed` do NLMR; `None` sorteia
e guarda em `attrs`), `rescale` e `n_threads`. Desempenho: o gerador é serial por natureza,
mas o que vem depois dele é paralelo (uniformes em série e `qnorm` em paralelo; passos do
diamond-square em paralelo; FFT 1D por linha/coluna; distância exata; rotulagem), e somem
os custos do R (o `vapply` por célula do `percolation`, o `dplyr` por linha do
`randomcluster`, as varreduras do `randomrectangularcluster`, o `disagg` do `curds`).

Referência: NLMR do clone local instalado numa biblioteca R própria
(`C:\Users\eduar\AppData\Local\R\landschaft-nlm-lib`, sem mexer na do RStudio), R 4.4.2,
`tools/make_reference_nlm.py` gravando em `tests/data/reference/nlm`; as distribuições da
libstdc++ conferidas com um programa compilado pelo g++ do Rtools.

Fases:
0. geradores do R e da libstdc++, biblioteca R com o NLMR, script de referência.
   **Feita** (`src/cpp/nlm/rng.cpp`: todos os fluxos bit a bit com o R 4.4.2 e com um
   programa compilado pelo g++ 13.3 do Rtools, na primeira tentativa);
1. gradientes, `random`, `percolation`, `rescale`, `classify` (com Jenks). **Feita**
   (0.54.0, tudo bit a bit com o NLMR). Descoberto: o `terra::rast(ncol, nrow, xmin = 0,
   xmax = ncol, ...)` sem CRS vira lon/lat quando o extent cabe em ±360 × ±90, então o
   `distancegradient` de rasters até 360 × 90 dá distâncias haversine em metros (esfera de
   6378137 m, busca podada por latitude do terra, `sin`/`cos`/`atan2` x87 do mingw); acima
   disso é o `GDALComputeProximity` (varredura com o alvo mais próximo propagado, float32).
   `sum`/`cumsum` das proporções do `classify` em 80 bits; o Jenks roda coluna a coluna
   com as linhas em paralelo (as mesmas somas);
2. `mpd`, `neigh`, `randomrectangularcluster`. **Feita** (0.55.0, bit a bit na primeira
   tentativa, também em toro e com o `shuffle` de mais de 65535 células; no `mpd` só o
   gerador e o teste de aceitação do método polar ficam em série, o `log`/`sqrt` e as
   células de cada passo vão em paralelo: 3× o NLMR em 1025²; `neigh` é serial por
   natureza);
3. `randomcluster`, `curds`. **Feita** (0.56.0, bit a bit; o `terra::patches` numera pela
   primeira célula de cada mancha em ordem de linha (a união guarda o menor rótulo, o R
   renumera em ordem); `randomcluster` 200² 19 s no R contra 3 ms);
4. o `fft` do R portado; `gaussianfield`, `fbm`. **Feita** (0.57.0, bit a bit com as
   `simulate_*` internas do NLMR; `fft.c` do R portado linha a linha, colunas e linhas em
   paralelo com os mesmos bits; `exp` e `x^y` do R no Windows são os do mingw-w64 v11 em
   x87 (`__expl_internal` com log2(e) partido em duas constantes; `powl`: expoente inteiro
   por quadrados em 80 bits, 0,5 por `fsqrt`, os demais `exp2l(y log2l |x|)`), agora em
   `lsm/rmath.hpp` (`r_exp`, `r_pow`) e no MASM; 6–8× o NLMR);
5. `mosaicfield` (`rpoislinetess`, `rMosaicField`, `rLGCP` do spatstat). **Feita** (0.58.0,
   bit a bit, também `collect` e `infinit`; `rpois`/`exp_rand` do R portados; os ladrilhos
   são as células do arranjo de retas no quadrado unitário na ordem lexicográfica do
   `chop.tess` (abaixo antes de acima), recortando o quadrado reta a reta (o polyclip do
   spatstat arredonda os vértices a ~1e-9: um pixel a essa distância de uma reta, ou uma
   lasca mais fina que isso, poderia divergir, nunca visto); 200² com 20 mosaicos: 9,95 s no
   R, 10 ms aqui);
6. `mosaictess`, `mosaicgibbs` (`st_sample`, `rSSI`, Voronoi). **Feita** (0.59.0, bit a bit, melhor
   que o previsto: a ordem das células do GEOS vem do próprio GEOS pelo shapely (3.13 dá a
   mesma ordem que o 3.12 do sf), dependência opcional com aviso; o `terra::rasterize` grava
   float32; a rasterização é por germe mais próximo em grade de baldes, em paralelo;
   25–70× o NLMR). Resta: portar a triangulação incremental do GEOS para não depender do
   shapely;
7. opcionais: `blend`/`merge` do antigo landscapetools, modelos do NLMpy, gerador paralelo
   rápido (`rng="fast"`, diferente do R) para rasters enormes. **Feita em parte** (0.60.0:
   `merge` e `binarize` do landscapetools (sem referência em R: o pacote saiu do CRAN), o
   tutorial *Neutral landscapes* com um teste de modelo nulo). Ficam de fora: o gerador
   paralelo (o gerador serial não é o gargalo: `random` 2000² leva 44 ms) e os modelos do
   NLMpy (outra referência, com o gerador do numpy); portar a triangulação do GEOS para
   o `mosaictess`/`mosaicgibbs` dispensarem o shapely.

## Análise de biodiversidade (Biodiverse)

Análise de 2026-09-28. O Biodiverse 5.99_004 (`C:\Users\eduar\Documents\dev\biodiverse`, Perl,
GPL-3, ~98 mil linhas em `lib/`, metade GUI Gtk) analisa a distribuição espacial da
biodiversidade. O núcleo:
- **BaseData**: matriz esparsa grupos × rótulos (células × espécies, contagem de amostras),
  grupos com coordenadas em N eixos e tamanhos de célula; propriedades de rótulos e grupos;
  importação de CSV, raster e shapefile;
- **condições espaciais**: expressões Perl avaliadas por `eval` (`sp_circle(radius => r)`,
  `sp_block`, `sp_annulus`, `sp_ellipse`, `sp_rectangle`, `sp_select_*`, `sp_is_left_of`,
  `sp_in_line_with`, `$D`, `$C`, `$d[0]`, `$coord[1]`...) definindo um ou dois conjuntos de
  vizinhos por grupo, e a *definition query*;
- **análise espacial**: 119 cálculos, 382 índices (`indices.json`): contagens e listas,
  endemismo, raridade, dissimilaridade taxonômica, estimadores de riqueza, rótulos
  hierárquicos, filogenéticos (PD, família PE, AED, clade contributions/loss, turnover,
  relativos RPD/RPE, PhyloCom MPD/MNTD/NRI/NTI, RWiBaLD), propriedades de rótulos e grupos
  (estatísticas, quantis, Gi*), rótulos numéricos, matriz e Rao QE;
- **randomizações** (`rand_structured`, `rand_csr_by_group`, independent swaps, curveball,
  random walk, diffusion, espacialmente estruturada) e comparações `C_/Q_/P_/SUM_/T_`,
  z-scores, significância e **CANAPE**;
- **cluster e RegionGrower**: aglomerativo sobre a matriz de dissimilaridade dos grupos, com
  as ligações do Biodiverse (average, average_unweighted, minimum, maximum, recalculate),
  tie-breakers, restrição espacial e índices por nó da árvore;
- árvores (Nexus/Newick) e matrizes de dissimilaridade entre rótulos.

Fora do escopo: a GUI, os arquivos de projeto `.bds`/`.bps` (Sereal), o Remap GUI e a
importação de planilhas (o pandas cobre).

Proposta: motor C++ `src/cpp/bio/` e subpacote `landschaft.bio` (`ls.bio`):

```python
bd   = ls.bio.BaseData.from_table("sites.csv", label_columns=["genus", "species"],
                                  group_columns=["x", "y"], cell_sizes=[100000, 100000])
tree = ls.bio.read_tree("tree.nex")
res  = ls.bio.spatial_analysis(bd, conditions=["sp_circle(radius=>200000)"],
                               calculations=["endemism_whole", "pd", "pe"], tree=tree)
res.table; res.to_raster("PE_WE")
rnd  = ls.bio.randomise(bd, res, function="rand_structured", iterations=999, seed=42)
ls.bio.canape(rnd)
cl   = ls.bio.cluster(bd, index="SORENSON", linkage="average"); cl.to_raster(n_clusters=10)
ls.bio.list_calculations()
```

- Nomes: índices com os nomes do Biodiverse (`PE_WE`, `ENDW_WE`), cálculos sem o prefixo
  `calc_` (aceito também com ele); metadados a partir do `indices.json` (GPL-3).
- Condições espaciais: parser da sintaxe do Biodiverse (funções `sp_*`, variáveis `$D`, `$C`,
  `$d[i]`, `$D[i]`, `$coord[i]`, `&&`, `||`, `!`, comparações, aritmética) numa AST avaliada em
  C++; callable Python como alternativa lenta.
- Desempenho: vizinhança por índice em grade/KD (círculo, quadrado, anel, elipse, retângulo),
  listas em CSR, OpenMP por grupo, reaproveitamento das vizinhanças de `sp_block`;
  "estatísticas suficientes" por vizinhança (acumulador denso por thread + rótulos tocados)
  alimentando todas as famílias de índices numa agregação; árvore em arrays em pós-ordem, PD
  subindo até a raiz com marcas, ranges dos nós pré-computados; A/B/C entre todos os pares de
  grupos por `X·Xᵀ` esparso (Eigen) ou GEMM em blocos; randomizações com arrays de inteiros,
  geração sequencial (mesmo fluxo do original) e índices por iteração em paralelo, com modo
  opcional de fluxos independentes para iterações em paralelo; `n_threads` em tudo.

Formatos modernos (o núcleo só recebe arrays grupos/rótulos/contagens; formatos são
adaptadores em Python, e a paridade é conferida pelo CSV):
- `from_dataframe` (tabela longa grupo/rótulo/contagem ou larga), `from_sparse`
  (scipy.sparse), `from_xarray`;
- ocorrências do GBIF / Darwin Core (DwC-A ou CSV simples: `decimalLatitude`,
  `decimalLongitude`, `species`/`taxonKey`, `individualCount`), com reprojeção e agregação em
  células e filtros (`coordinateUncertaintyInMeters`, duplicatas, `basisOfRecord`);
- pilhas de SDM (GeoTIFF/COG multibanda, NetCDF, Zarr; uma banda/variável por espécie, limiar
  ou peso por probabilidade), lidas em blocos (dask) direto para a matriz esparsa;
- GeoParquet e GeoPackage na entrada e na saída (resultados com a geometria das células), COG,
  e o CSV no formato do Biodiverse;
- CRS explícito no `BaseData` (unidades de `sp_circle` claras);
- grades hexagonais e H3 (`h3` opcional), hexágonos do `make_grid`;
- árvores: Newick/Nexus com parser próprio; objetos `Bio.Phylo`/`dendropy`/`ete` se
  instalados;
- formato nativo de sessão em Parquet/Zarr em vez do Sereal.
Dependências novas só em extras opcionais (`h3`, `pygbif`, `dendropy`); `pyarrow` e
`geopandas` já estão no extra `vector`.

Critério de igualdade (decidido 2026-09-28): o Strawberry Perl usa `double` (sem long double
nem x87). O Biodiverse soma vários valores na ordem do hash (`sum values %$nodes_in_path` no
PD), que muda o último bit a cada execução. Referências com `PERL_HASH_SEED=0` e
`PERL_PERTURB_KEYS=0` e um overlay no harness que troca essas somas pela ordem das chaves
ordenadas; do nosso lado os ids de rótulos e nós seguem a ordem das strings, e a comparação é
**bit a bit** contra o Biodiverse com o overlay, ~1e-12 contra o original sem patch. PRNG:
`Math::Random::MT::Auto` (MT19937-64, `irand`, `rand`, `shuffle`, `binomial`, `gaussian`)
portado do C; percentis de `Statistics::Descriptive`/PDL portados. Fixtures extras: os
valores esperados embutidos em `biodiverse/t/24-Indices-*.t` (`__DATA__`). Checagem cruzada
do CANAPE com o pacote R `canaper`.

Referência: Strawberry Perl 5.42 (`C:\Strawberry`), dependências do Biodiverse com
`cpanm -L` numa biblioteca própria (sem mexer na do Strawberry); shims no harness para o que
não for preciso (GDAL), já que a referência lê CSV; `tools/make_reference_bio.py` +
`tools/bio_reference.pl` gravando em `tests/data/reference/bio`.

Fases:
0. harness Perl (biblioteca própria, overlay de soma ordenada, script de referência), dados
   de exemplo. **Feita** (0.61.0): `tools/bio_harness.py` monta o overlay reescrevendo
   todo `keys`/`values`/`each`, `for my ($k, $v) (%h)`, `pairs`/`pairmap` sobre hashes
   (1100+ pontos) para funções `LSK::*` que devolvem as chaves ordenadas respeitando o
   contexto escalar, e o `foreach (values %h)` para uma fatia ordenada (mantém o alias);
   `tools/bio_reference.pl` roda o Biodiverse sobre um job JSON e grava números com 17
   dígitos. Biblioteca Perl em `%LOCALAPPDATA%/landschaft-bio-perl5` (`cpanm -L`, PDL
   compilado); GDAL pelo `Geo::GDAL::FFI` apontado para o `gdal.dll` do ambiente conda
   `geo` (módulo `Geo::GDAL::gdal` e um `Alien::gdal` de fachada); GSL 2.8 pelo
   `Alien::GSL` (compilado com o MSYS) e `PDL::GSL` com um `bin/gsl-config.bat` que
   responde versão e flags;
1. `BaseData` (tabela, DataFrame, esparsa, xarray, pontos, raster, vetor, GeoParquet/GPKG,
   CRS, `to_raster`, ranges, contagens), `Tree` (Nexus/Newick), `Matrix`. **Feita**
   (0.61.0): tudo igual ao Biodiverse bit a bit (grupos, coordenadas, contagens, ranges,
   árvores com os nomes `N___`/`__dupN`, matrizes), menos as áreas de polígonos (7e-15: o
   Biodiverse corta primeiro numa grade 8× mais grossa). Descobertas: o Biodiverse pede
   `quote_space => 0` ao Text::CSV_XS mas filtra a opção, então nomes com espaço ou tab
   saem entre aspas (`'east west':11.25`); contagem vazia vira 0 (`// 0` antes do teste de
   indefinido); a importação de texto arredonda `(x - origem) / tamanho` a 1e-10 antes do
   `floor`, a de feições e a de raster não; números viram texto com `%.15g` e as
   coordenadas são relidas do nome;
1b. GBIF/Darwin Core e pilhas de SDM (em blocos). **Feita** (0.62.0): `read_occurrences`
   (download "simple CSV" do GBIF em zip ou tsv, Darwin Core Archive pelo `meta.xml`, ou
   DataFrame do `pygbif`; lido em pedaços; filtros de ausência, coordenadas, incerteza,
   `basisOfRecord`, issues e duplicatas) e `BaseData.from_occurrences` (reprojeção pelo
   `rasterio.warp`, células como na importação de texto); `from_raster` e `from_xarray`
   leem em blocos de linhas (COG, NetCDF/Zarr com dask); `_from_records` trabalha com
   códigos inteiros e só nomeia os grupos distintos. Nomes com espaço: a importação de
   texto põe aspas como o Biodiverse; as de ocorrências e de pilhas mantêm o nome puro;
2. motor de vizinhança: parser das condições espaciais, todas as `sp_*`, *definition query*,
   reaproveitamento de vizinhanças. **Feita** (0.63.0): `ls.bio.neighbours`,
   `definition_query`, `SpatialCondition`; parser de Perl (`_perlexpr.py`, precedência do
   Perl, `my`, `qr//`, `q{}`) compilado para uma árvore avaliada em C++ por par de grupos
   (`src/cpp/bio/conditions.cpp`, grade uniforme nos dois primeiros eixos quando a condição
   limita a distância, OpenMP); as funções que só dependem de um grupo (texto, regex,
   polígonos, riqueza) viram máscaras em Python. 48 casos iguais ao Biodiverse, inclusive as
   peculiaridades: o "tipo de resultado" é deduzido do texto (junta os tipos das `sp_*`
   chamadas: `sp_select_all() && $D < 5` pega todos os grupos, `sp_point_in_poly(...) && $D
   < 2` usa a lista do primeiro grupo para todos) e as listas de vizinhos são reaproveitadas
   entre grupos (`non_overlapping` só no primeiro conjunto, `text_match_exact`,
   `always_true`, `always_same`), emulado em Python só quando a condição mistura essas
   funções com termos que dependem do local. Descobertas: o Perl do Strawberry usa a mesma
   libm x87 do mingw que o R (`lsm/rmath.hpp`: sin, cos, exp, log, atan2 batem 100%); o `**`
   do Perl eleva a expoentes inteiros por quadrados em double (negativos: 1/x^n) e usa o
   `powl` x87 nos demais (`bio/perlmath.hpp`); `$C[i]` sozinho fica indefinido (o
   Biodiverse só calcula distâncias em células com `$C` ou `$c[i]`); `WHERE "campo" =
   "2"` do SQLite compara número. 490 mil grupos, círculo de raio 3: 0,5 s. Ficam para as
   fases seguintes as condições de ranges de rótulos, clusters, saídas espaciais e o
   `sp_select_block` (aleatório);
3. análise espacial e índices não filogenéticos: contagens e listas, endemismo, raridade,
   dissimilaridade taxonômica, estimadores de riqueza, rótulos hierárquicos. **Feita**
   (0.64.0): `ls.bio.spatial_analysis`, `list_calculations`, `SpatialResult` (tabela de
   escalares, listas longas construídas sob demanda, `to_raster`); motor em C++
   (`src/cpp/bio/indices.cpp`): agregação ABC por vizinhança (presença, grupos por rótulo,
   soma das amostras grupo a grupo na ordem da lista, o conjunto 1 somado antes do 2) e os
   45 cálculos portados linha a linha, com OpenMP por grupo. 11 análises (uma e duas
   vizinhanças, blocos, *definition query*, contagens inteiras e fracionárias) iguais bit a
   bit ao Biodiverse. Descobertas: o `sqrt` do Perl é o x87 do mingw (`r_sqrt`, arredondamento
   duplo: 1 ulp em desvios padrão e erros do Chao/ACE); `Statistics::Descriptive` soma com
   `List::Util::sum`, variância `(Σx² − n·média²)/(n−1)` limitada a 0, percentil pelo índice
   `int(0.5 + (n−1)p/100)`; hashes de chaves numéricas (frequências nos estimadores) são
   percorridos na ordem das chaves como texto; `calc_hierarchical_label_ratios` do Biodiverse
   nunca usa as bases por nível que monta (todos os níveis dão o mesmo A/B/C), reproduzido; o
   ACE divide por zero (e o Biodiverse aborta) quando `a2 == 1`. Grupos com as mesmas
   vizinhanças são calculados uma vez quando nada depende do grupo processado (blocos:
   19 s → 4 s em 90 mil grupos com todos os cálculos; o custo restante é o volume das
   listas, ~180 milhões de entradas);
4. filogenéticos: PD, família PE, clade contributions/loss, AED, PD local, turnover
   filogenético. **Feita** (0.65.0): 33 cálculos com `tree=` (`src/cpp/bio/phylo.cpp`:
   árvore em arrays na ordem dos nomes, alcance dos nós na árvore podada por marcação dos
   caminhos grupo a grupo, PE por grupo calculado uma vez, ES/ED/AED globais; por vizinhança,
   uniões de caminhos com marcas e a sub-árvore de `get_sub_tree_as_hash` com os filhos na
   ordem de inserção). 7 análises iguais bit a bit ao Biodiverse, também com poda e com
   rótulos fora da árvore. Descobertas: o `trim` do Biodiverse só poda (termina com a remoção
   dos nós `___` sem descendente nomeado, e só se algo foi removido antes); o PE_WE_P usa o
   comprimento da árvore podada e o PD_P o da árvore inteira; `_calc_abc_any` repassa o
   primeiro ABC calculado na ordem dos pré-cálculos (presença, grupos ou amostras), o que
   muda os valores de `PHYLO_LABELS_NOT_ON_TREE`: a ordem é reproduzida em Python a partir
   das dependências extraídas do Biodiverse (`bio/data/biodiverse_dependencies.json`); as
   contribuições de clado truncam em 1e-11 (`int(1e11 x) / 1e11`), por isso o Biodiverse
   sem o overlay varia nesse dígito. PD e PE em 90 mil grupos (árvore de 500 pontas): 0,5 s.
   O RWiBaLD depende da árvore de ramos equalizados e fica na fase 5;
5. filogenéticos relativos (RPD/RPE, árvores aparadas), PhyloCom (MPD/MNTD, NRI/NTI, valores
   esperados, VPD), RWiBaLD, CWE/CWR. **Feita** (0.66.0): RPD1/2, RPE1/2/central, rótulos na
   árvore aparada e RWiBaLD (limiares de cotovelo do PDL) iguais bit a bit nos 7 casos da fase 4;
   PhyloCom (`src/cpp/bio/phylocom.cpp`) em 8 casos novos, bit a bit, inclusive NTI e VPD
   amostrados. Descobertas: o cache de caminhos do Biodiverse depende da ordem dos grupos em
   árvores ultramétricas (o comprimento é o dobro do caminho do primeiro rótulo do primeiro par
   consultado sob cada ancestral, e só os filhos vizinhos por número de terminais recebem o
   valor): em árvores binárias o valor de cada ancestral fica fixo no primeiro grupo que o
   abrange, então uma passagem sequencial barata o determina e o resto roda em paralelo; com
   politomias (ou rótulos em nós internos) o processamento é sequencial e grupos com a mesma
   vizinhança não são deduplicados. O NTI amostrado usa o Math::Random::MT::Auto (MT19937-64,
   portado e conferido com o Perl) num único fluxo na ordem de primeira aparição de cada número
   de rótulos; com VPD, dois rótulos refazem a amostragem a cada grupo (consumindo o gerador) e
   isso é reproduzido. Os sorteios são feitos em lotes: conjuntos em sequência, distâncias em
   paralelo, e o gerador volta ao ponto onde o Biodiverse para. 90 mil grupos, árvore de 500
   pontas: MPD+MNTD 2,2 s, NRI/NTI 1,1 s (ultramétrica) e 2,0 s (amostrado);
6. propriedades de rótulos e grupos (estatísticas, quantis, Gi*), rótulos numéricos, índices
   de matriz, Rao QE. **Feita** (0.67.0): 29 cálculos (`src/cpp/bio/props.cpp`), 9 análises
   iguais bit a bit. Descobertas: as estatísticas vêm de um objeto
   Statistics::Descriptive::PDL::SampleWeighted compartilhado pelos cálculos do grupo, com
   cache e uma ordenação feita na primeira mediana ou quantil, depois da qual as somas seguem a
   ordem ordenada: o resultado depende da ordem em que o Biodiverse roda os cálculos (a dos
   pré-cálculos), reproduzida; os valores são as chaves de um hash (a string lida, então
   "20.700" e "20.7" são valores distintos) com pesos inteiros truncados; o PDL soma a partir
   de 0, eleva a potências inteiras por quadrados e tira raiz com arredondamento correto (o
   `sqrt` do Perl é x87); o Gi* usa o `sqrt` do Perl. O overlay do harness passou a reescrever
   também os módulos do Statistics::Descriptive::PDL (a ordem das chaves dentro deles chega
   aos resultados; as referências anteriores não mudaram). Campos vazios nos arquivos de
   propriedades ficam definidos sem valor (listados, fora das estatísticas); valores não
   numéricos são descartados. 90 mil grupos: 0,3 a 0,6 s;
7. MT::Auto portado, randomizações, comparações, z-scores, significância, CANAPE.
   **7a feita** (0.68.0): `ls.bio.randomise` com `rand_csr_by_group` e `rand_nochange`,
   contadores C/Q/T/P/SUMX/SUMXX por índice e por item das listas, p-ranks, z-scores (com o
   `sqrt` x87 do Perl) e as quatro variantes do CANAPE, iguais bit a bit (3 casos, com
   reciclagem de resultados). Descobertas: o Biodiverse descarta índices que pedem mais
   conjuntos de vizinhança que os dados (`index_uses_nbr_lists`, agora extraído com as
   dependências por `make_reference_bio.py dependencies`), então `RICHNESS_SET2` etc. não
   aparecem com um conjunto; o `RECYCLED_SET` interno entra nas comparações e é ignorado.
   A análise é refeita pelo `_evaluate` com as vizinhanças originais. **7b feita** (0.69.0):
   `rand_structured` em C++ (`src/cpp/bio/randomise.cpp`) com a fase de trocas
   (`swap_to_reach_richness_targets`, incluindo o List::Unique::DeterministicOrder e suas
   remoções por troca com o último), multiplicador e adição de riqueza; 4 casos bit a bit.
   Peculiaridades reproduzidas: `scalar @$obj` num DeterministicOrder vale sempre 2 (as
   remoções "se vazio" nunca ocorrem) e um `delete` remove um rótulo de uma lista de grupos.
   A escolha do j-ésimo grupo ainda livre usa uma árvore de Fenwick (0,95 s por
   randomização com 90 mil grupos e 630 mil registros). **7c feita** (0.70.0):
   `rand_independent_swaps`, `rand_independent_swaps_modified` (com o
   Statistics::Sampler::Multinomial::Indexed portado) e `rand_curveball` (desvio binomial do
   MT::Auto portado e conferido com o Perl nos três caminhos), com `swap_count`,
   `max_swap_attempts`, `stop_on_all_swapped` e `use_max_swap`; 6 casos bit a bit. Na versão
   modificada o Biodiverse nunca preenche os pares originais, então `stop_on_all_swapped` não
   tem efeito, como lá. Ficam para depois: o modo hipergeométrico do curveball (desvio
   gaussiano e CDF do GSL) e os pares de troca espaciais. **7d feita** (0.71.0): alocação
   espacial do `rand_structured` (`rand_spatially_structured`, `rand_diffusion`,
   `rand_random_walk`: ordens difusão, passeio aleatório, aleatória e proximidade,
   retrocesso, re-semeadura, condição de semeadura), randomização por subconjuntos
   (`spatial_conditions_for_subset`, `definition_query`), rótulos fixos
   (`labels_not_to_randomise`), propriedades de grupo (`by_set`, `by_item`) e embaralhamento
   dos nomes terminais das árvores; 17 casos novos bit a bit (32 no total). Peculiaridades
   reproduzidas: os caches de vizinhança e o sorteio de desempate da proximidade valem para
   todas as iterações; os alvos da condição de semeadura só são usados na primeira (o cache
   por rótulo evita recalculá-los); uma semente que já saiu da lista escolhe o último alvo
   (índice -1); subconjuntos usam um objeto de randomização novo por iteração; os grupos
   reprovados pela definição mantêm as propriedades embaralhadas da iteração anterior; com
   rótulos fixos a base randomizada perde as propriedades. Ficam de fora: condições que
   dependem do rótulo corrente, `track_label_allocation_order`, e os itens do 7c;
8. cluster e RegionGrower (ligações, tie-breakers, restrição espacial, índices por nó, export
   Newick/Nexus). **Feita** (0.72.0): `ls.bio.cluster` e
   `ls.bio.region_grower` (`src/cpp/bio/cluster.cpp`: a matriz do Biodiverse com o índice por
   valor em `%.2g` e empates pela string `%.15g`, a construção das matrizes por condição com a
   matriz-sombra, o laço de junções com as ligações média, média não ponderada, mínima, máxima
   e recálculo, empates por embaralhamento, por `random` ou agrupando zeros; a árvore termina
   em Python como o TreeNode: comprimentos pelos valores de junção, ramos de comprimento zero
   subidos, raízes unidas, `ladderise` e numeração dos terminais). 14 análises iguais bit a bit,
   inclusive os índices por nó. Descobertas: no modo par a par o PHYLO_B/C vem do total menos o
   compartilhado; no modo hierárquico os nós herdam as contagens do primeiro filho e os
   rótulos dos outros filhos recebem contagem 1 (as amostras são somadas), o que muda o
   endemismo por nó; a análise de vizinhança exclui os grupos já processados. Ficam para depois:
   desempates por índice, `link_recalculate` com cache de ABC e a exportação; 2500 grupos em
   13 s;

9. grades hexagonais/H3, formato nativo de sessão, árvores de `Bio.Phylo`/`dendropy`/`ete`.
   **Feita** (0.73.0): `BaseData.from_hex` (hexágonos com ponta ou lado para cima, grupos nos
   centros para as condições por distância) e `BaseData.from_h3` (`h3` opcional, centros em
   graus, ids das células em `grid`), desenhados como hexágonos pelo `to_geodataframe`;
   `BaseData.save`/`load` e `save_session`/`load_session` (zip de Parquet: base, árvores em
   Newick, matrizes e resultados espaciais e de agrupamento); conversão de árvores para e de
   Bio.Phylo, DendroPy e ete pelo Newick com precisão total (o escritor do Bio.Phylo arredonda
   os comprimentos) lido pelo nosso parser compatível com o Biodiverse. Extra opcional `bio`;
10. extras (IEI do BiodiverseX, export no formato do Biodiverse), tutorial, otimização
    (janela incremental etc.). **Feita** (0.74.0): `iei_stats` e `iei_data` (o plugin
    BiodiverseX::Indices::IEI entra no `@ISA` do harness), iguais bit a bit;
    `SpatialResult.to_biodiverse_csv` escreve o texto do export "Delimited text" byte a byte
    (9 casos: listas simétricas e assimétricas, listas-array contadas, `no_data_value`,
    separador tab e aspas simples; `make_reference_bio.py export`); `ClusterResult.cut(k)`;
    `ls.bio.example_data` (dados de exemplo do Biodiverse no pacote) e o tutorial
    `docs/tutorials/biodiversity.md`. Descobertas: o export simétrico pega as chaves do
    primeiro elemento e troca indefinidos por `no_data_value`; o assimétrico usa a união das
    chaves, deixa vazios os indefinidos e soma 1 ao `no_data_value` por ocorrência nas
    listas-array; as chaves seguem o `natsort` do Sort::Key::Natural (dígitos pelo
    comprimento). Otimização: a janela incremental ficou desnecessária (17,6 mil grupos com
    PD/PE/RPE em 0,1 s); o gargalo real era a alocação espacial das randomizações
    (conjuntos por rótulo em `std::set` percorridos para cada ocorrência → vetores
    marcados por passagem: `rand_diffusion` 10× e `rand_random_walk` 5× mais rápidos, com as
    listas de vizinhos montadas em numpy) e as iterações da família `rand_structured` passam a
    randomizar a próxima enquanto a atual é analisada (~30 %; as trocas ficam em sequência
    porque o curveball perde velocidade ao lado das threads da análise).

## Distâncias e rotas em grades (gdistance)

Análise de 2026-09-29. O gdistance 1.6.4 (`C:\Users\eduar\Documents\dev\gdistance`, R,
GPL >= 2; na biblioteca do RStudio está o 1.6.5, que só muda a documentação) calcula
distâncias e rotas sobre o grafo de células vizinhas de um raster: a matriz de transição
(`transition`, com uma função dos dois valores, 4/8/16 direções, simétrica ou não;
`geoCorrection`; `normalize`; aritmética; `sumReciprocal`; barreiras, áreas e Mahalanobis),
o menor custo (`accCost`, `costDistance`, `shortestPath`), o passeio aleatório
(`commuteDistance`, `passage` sem `theta`), os passeios aleatórios de menor custo (RSP:
`passage` com `theta`, `rSPDistance`) e a sobreposição de rotas (`pathInc`).

O que já existia no landschaft: o grafo de condutância só por dentro do `conn` (duas funções,
4/8 direções, simétrico, raster projetado), o `costDistance` e o `commuteDistance` só entre
os nós de um `HabitatGraph`, e as versões do GTB (`ls.cost_distance`, `ls.least_cost_path`),
com outra semântica, que ficam como estão.

Desenho (aceito pelo usuário): submódulo `ls.cost` (`src/cpp/cost/`), nomes em snake_case.
- `Transition`: um valor por célula e por direção (`values[d, célula]`; metade das direções
  quando simétrica, o valor de `f(x[menor índice], x[maior índice])`; todas quando não, o
  valor de `f(x[origem], x[destino])`), 0 = sem ligação. Mais compacto e previsível que uma
  matriz esparsa genérica; lon/lat global dá a volta no meridiano 180 como o gdistance.
- Funções prontas em C++ (`"mean"` com o `mean` do R, min, max, soma, origem, destino...)
  ou qualquer callable vetorizado que recebe `x` (2 x n) como a função do R recebe `c(x1, x2)`;
  `1 / tr`, `tr * tr2`, `tr.map(f, pairs="edges"|"all")` para o exemplo de Tobler.
- Pontos como coordenadas (n x 2), pontos do shapely/GeoDataFrame, ou máscaras raster.

Critério de igualdade (referências geradas no R com `tools/make_reference_cost.py`, casos em
`tests/cost_cases.py`): bit a bit na transição, na geoCorrection (tentar também lon/lat:
GeographicLib 2.0 do pacote raster com a libm do mingw), no normalize, na aritmética, no
`acc_cost` e no `cost_distance`; o `shortest_path` com a mesma sequência de células,
inclusive nos empates (ordem do heap do Dijkstra do igraph); commute, `passage`, RSP e
`path_inc` com tolerância (~1e-10), porque os solvers esparsos somam em outra ordem.

Fases:
0. harness de referência, casos, dados de exemplo (`maungawhau`, `Europe`, `genDist`,
   `popCoord`) e o esqueleto de `ls.cost`;
1. `Transition` (motor C++/OpenMP, funções prontas e callables, 4/8/16 direções, assimétrica,
   `geo_correction` c/r com lon/lat e volta global, `normalize`, aritmética,
   `sum_reciprocal`, `to_raster`, `adjacency`, Mahalanobis);
2. menor custo: `acc_cost`, `cost_distance` (from/to, assimétrico), `shortest_path`
   (células, raster, transição, linhas), Dijkstra direto na grade, origens em paralelo;
3. passeio aleatório: `commute_distance` e `passage` sem `theta` com a Cholesky multifrontal
   do `circuit` (o `HabitatGraph` passa a usá-la no lugar do SuperLU);
4. RSP: `passage(theta=...)` e `rsp_distance` (métodos 1/2, net/total); uma fatoração de
   `I - W` e Sherman-Morrison por destino (`I - W_j` difere em uma linha) no lugar de uma
   fatoração por par;
5. `path_inc` (overlap/nonoverlap, pesos, barreiras e áreas), tutorial com figuras e
   benchmarks contra o gdistance. O `pathInc` com `to=` do gdistance devolve só NA (o
   `.finishFlowFromTo` copia a matriz vazia): implementar o comportamento pretendido;
6. opcional, se o profiling justificar: LU multifrontal própria sem pivotamento para o RSP
   (`I - W` é diagonal dominante), com a ordenação ND do `circuit`.

Fora (não faz sentido portar 1:1): a estrutura S4 (`Coords`, `TransitionData`, `show`,
`==`, `[<-` com matriz de índices, `Summary` linha a linha), a função do R chamada ligação
por ligação (aqui vetorizada), o processamento em pedaços em arquivo temporário do
`pathInc` e as duas rotinas internas de transição (mesmos valores).

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
