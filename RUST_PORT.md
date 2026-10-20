# Transição do C++ para o Rust

Registro de 2026-10-20. O plano de portar o núcleo compilado do landschaft (`src/cpp`) de C++ para Rust, aos
poucos, módulo por módulo, sem que o landschaft deixe de funcionar ou de bater com as referências em nenhum
momento. O Terraflux cita este arquivo (`terraflux/IDEAS.md`, "Todo em Rust").

## Objetivo

1. **O Terraflux lê o núcleo em Rust diretamente.** Hoje ele compila o C++ do landschaft pelo CMake
   (`terraflux-ls-sys`, `LANDSCHAFT_PYTHON=OFF`) e o chama por um shim do `cxx` (cerca de 8,8 mil linhas entre
   `shim.cc`, `shim.h` e `lib.rs`, com callbacks do C++ para o Rust). No fim, os crates do landschaft são
   dependências Rust comuns do Terraflux, o `terraflux-ls-sys` e o `cxx` deixam de existir, e o build é só
   `cargo`.
2. **O Terraflux no navegador (WebAssembly) fica simples.** Com o núcleo em Rust, o alvo
   `wasm32-unknown-unknown` passa a valer para o motor e para a interface egui, sem o Emscripten, o `emcc`, o
   `-fwasm-exceptions` e as pontes por funções C que o caminho "C++ pelo Emscripten" exigiria.
3. **Nada piora para quem usa o landschaft pelo Python.** A API Python não muda. Os resultados continuam batendo
   bit a bit com as referências (R, Perl, GTB, Julia, Whitebox, GDAL, etc.) e com o C++ atual, e a velocidade
   não cai.

## Onde estamos

Cerca de 100 mil linhas de C++ em `src/cpp`, incluindo as camadas do pybind11 (`bindings.cpp` e os
`python.cpp` de cada módulo). Por cima delas ficam cerca de 86 mil linhas de Python em `src/landschaft`, que não
são reescritas.

| Módulo | Linhas | Bibliotecas externas | `#pragma omp` | Depende de |
|---|---:|---|---:|---|
| `swat` | 28 885 | — | 0 | — |
| `opt` | 9 778 | HiGHS | 15 | `lsm` |
| `lsm` | 7 693 | — (x87 por MASM / asm inline) | 42 | — |
| `bio` | 7 563 | — (matemática do Perl) | 8 | `lsm` |
| `gtb` | 7 006 | — | 33 | — |
| `sim` | 5 995 | Eigen (denso, complexo) | 3 | `lsm`, `nlm` |
| `sdm` | 4 504 | GeographicLib (vendorizado) | 12 | `lsm`, `nlm`, `cost` |
| `circuit` | 3 930 | Eigen (denso, Cholesky multifrontal) | 24 | — |
| `nlm` | 3 773 | — (geradores do R e da libstdc++, FFT do R) | 0 | `lsm` |
| `surf` | 3 707 | — | 29 | `lsm`, `nlm`, `cost` |
| `io` | 3 367 | earcut e exactextract (vendorizados) | 30 | `lsm` |
| `cost` | 2 990 | Eigen (esparso, `SparseLU`) | 28 | `lsm`, `circuit` |
| `prio` | 2 912 | Eigen | 38 | — |
| `conn` | 1 909 | — | 2 | `lsm` |
| `hydro` | 1 908 | — | 2 | — |
| `rest` | 1 020 | — | 18 | `lsm` |
| `serv` | 930 | — | 12 | — |

Duas coisas fazem do `lsm` a base de quase tudo. A primeira é `lsm/rmath.hpp`: `log`, `sqrt`, `exp`, `sin`,
`cos`, `atan2` e `pow` calculados na unidade x87 em precisão estendida, como o R no Windows. A segunda é
`lsm/extended.hpp`: o `long double` de 80 bits do R, emulado com inteiros de 128 bits onde o compilador não o
tem.

O Terraflux chama quase todos os módulos pelo shim (`surf`, `opt`, `cost`, `sim`, `gtb`, `conn`, `circuit`,
`bio`, `nlm`, `lsm`, `prio`, `sdm`, `hydro`, `serv`, `io`, `rest`, `swat`).

O GDAL e o PROJ não estão no C++ do landschaft. No Python eles vêm do rasterio (e do pyproj, por baixo do
geopandas). No Terraflux, o crate `terraflux-gdal` carrega o GDAL em tempo de execução, e o próprio Terraflux lê
e escreve GeoTIFF quando o GDAL falta. O warp de `ls.reproject` (`io/warp.cpp`) é um port do kernel de
reamostragem do GDAL e pode ser portado para Rust como os demais.

## Regras da transição

1. **Um módulo por vez, inteiro.** Um módulo C++ só é apagado quando o Rust o substitui por completo: todos os
   kernels, todos os testes, os mesmos números e a mesma velocidade. Enquanto isso, os dois convivem no mesmo
   wheel.
2. **O C++ para de crescer.** A partir desta transição, lógica nova ou portada do Python vai para Rust, nunca
   para C++. Isso inclui a regra do `terraflux/docs/PLAN.md`, "a lógica que só existe em Python é portada para
   C++ dentro do landschaft", que passa a dizer Rust. Uma correção de bug num módulo ainda em C++ é feita no C++
   e, se o módulo já estiver sendo portado, também no Rust, com um teste que pegue o bug nos dois.
3. **Paridade bit a bit com o C++** (seção "Paridade"). Uma diferença só é aceita com uma explicação escrita
   aqui (seção "Diferenças aceitas") e um teste com a tolerância declarada.
4. **Desempenho igual ou melhor** (seção "Desempenho"). O Rust não substitui o C++ se estiver mais lento.
5. **Ferramentas equivalentes do Rust** para o que o C++ usa (OpenMP, Eigen, etc.). Onde não houver equivalente
   (GDAL, PROJ, HiGHS), usa-se o melhor que existir, e as perdas ficam registradas aqui.
6. **Cada crate portado compila para `wasm32-unknown-unknown` desde o primeiro dia**, conferido no CI. Um crate
   que só compila nativo atrasa o objetivo principal.
7. **Cada fase termina com um release** (versão minor, `CHANGELOG.md`), commit e push, e o progresso é anotado
   aqui.

## Arquitetura durante a transição

```
landschaft/
  Cargo.toml                 workspace
  rust/
    ls-math/                 x87 (asm!) e 80 bits por software, geradores do R e da libstdc++, heap e sort "do C++"
    ls-par/                  paralelismo determinístico sobre o rayon (blocos fixos, reduções em ordem fixa)
    ls-serv/, ls-hydro/, ... um crate por módulo, sem PyO3 e sem I/O: só fatias, arrays e structs
    ls-py/                   cdylib PyO3 + numpy: o módulo Python landschaft._rcore
  src/cpp/                   o C++ que ainda não foi portado (encolhe a cada fase)
  src/landschaft/            o Python, que escolhe _core (C++) ou _rcore (Rust) por kernel
```

- **Crates de kernel sem dependência de Python.** Recebem `&[f64]`, `&mut [u8]`, dimensões e parâmetros, e
  devolvem `Result<_, Error>`. São os mesmos crates que o Terraflux usa e que vão para o WASM.
- **Python:** PyO3 + `numpy` (rust-numpy), com o GIL liberado nos cálculos (`py.allow_threads`), arrays sem
  cópia quando são contíguos e erros convertidos nas mesmas exceções que o pybind11 lança hoje (`ValueError`,
  `MemoryError`, `RuntimeError`).
- **Build do wheel durante a transição:** o `scikit-build-core` e o CMake continuam. O CMake compila o crate
  `ls-py` pelo [Corrosion](https://github.com/corrosion-rs/corrosion), e o wheel leva `_core` (C++) e `_rcore`
  (Rust). Quando o C++ acabar, o backend passa a ser o `maturin`.
- **Seleção do motor:** cada wrapper Python pede o kernel a um registro (`landschaft._engine`). A variável
  `LANDSCHAFT_ENGINE=cpp|rust` força um dos dois em todos os kernels que existem nos dois motores, para rodar a
  suíte inteira com cada um. O padrão é o Rust onde ele já passou nas duas condições (paridade e desempenho).
- **Terraflux:** o `extern/landschaft` (submódulo) já traz os crates, e o Terraflux depende deles por `path`. A
  cada fase, as funções do módulo portado saem do shim do `terraflux-ls-sys` e passam a ser chamadas direto no
  crate, com os testes do Terraflux iguais. Os callbacks do C++ para o Rust (por exemplo o `link_recalculate`)
  somem quando o módulo que os usa é portado.

## Equivalentes no Rust

| C++ hoje | Rust | Observações |
|---|---|---|
| OpenMP (`#pragma omp parallel for`, `reduction`, `schedule`) | `rayon`, por baixo do `ls-par` | O C++ já divide o trabalho em blocos fixos para o resultado não depender do número de threads. O `ls-par` preserva isso: `par_chunks` de tamanho fixo e somas parciais combinadas em ordem fixa. As reduções do `rayon` (`sum`, `reduce`) não têm árvore fixa e não podem ser usadas em ponto flutuante. O `threads=` do Python vira um `ThreadPool` com escopo. |
| `omp.h` em tempo de execução (vcomp, libomp do conda-forge no macOS, gomp) | nenhum | O wheel deixa de levar o runtime do OpenMP. |
| Eigen denso (`MatrixXd`, `LLT`, `Map`, `OuterStride`) | `faer` para o que for novo; para paridade, port dos kernels exatos do Eigen usados | A ordem das somas do Eigen depende do bloqueio e do tamanho do pacote SIMD (por isso existe o `_circuit_avx2`). `faer` e `nalgebra` somam em outra ordem. Bit a bit, só portando a mesma ordem. |
| Eigen esparso (`SparseMatrix`, `SparseLU`) | port do algoritmo usado (`cost`); `faer::sparse` se a paridade não for exigida | O mesmo problema de ordem, mais a ordenação de colunas (COLAMD) do Eigen, que precisa ser a mesma. |
| Eigen complexo (`MatrixXcd`, autovalores em `sim`) | port do `EigenSolver` (Hessenberg + QR de Francis) usado | A raiz da matriz de Markov depende dos autovetores: o port precisa ser o mesmo algoritmo. |
| HiGHS (C++, compilado pelo CMake, com `highs_patch.cmake`) | `highs-sys` / `highs` (bindings para o HiGHS em C++) | Não há MILP em Rust à altura (`good_lp` com `microlp` é ordens de grandeza mais lento). O HiGHS continua C++, compilado pelo `cc`/`cmake` do `highs-sys`, com o patch aplicado a uma cópia vendorizada. |
| pybind11 | PyO3 + `numpy` | Com `abi3`, talvez um wheel por plataforma para todos os Pythons ≥ 3.10, em vez de cinco (confirmar com o rust-numpy). |
| x87 por MASM (MSVC) e asm inline (gcc/clang) | `core::arch::asm!` em x86-64 | `asm!` é estável e roda em todos os alvos x86-64, inclusive o MSVC, então o arquivo MASM some. |
| `long double` de 80 bits / `SoftExtended` | port direto do `SoftExtended` com `u128` | O Rust não tem `f80`. A emulação já existe e já dá os bits do R. |
| `std::mt19937`, distribuições da libstdc++, gerador do R | port à mão em `ls-math` (o motor `rand_mt` serve para o MT19937) | As distribuições do crate `rand` não dão os bits do R nem da libstdc++. |
| `std::sort`, `std::nth_element`, `std::priority_queue` (121 usos) | `sort_by` com ordem total ou port do algoritmo da STL | Onde a ordem dos empates muda o resultado, porta-se o introsort e o heap da STL usada nas referências. Onde não muda, uma ordem total explícita torna o resultado independente do algoritmo. |
| `std::exp`, `std::log`, `std::atan`, ... (CRT) | `f64::exp`, `f64::ln`, ... | Em código nativo o Rust chama a mesma libm da plataforma (UCRT no Windows, glibc no Linux), então os bits são os mesmos do C++. No WASM é o `libm` (port da musl): outros últimos bits. |
| `/fp:precise`, `-ffp-contract=off` | padrão do Rust | O Rust nunca funde em FMA nem reordena ponto flutuante por conta própria. Fica mais simples que hoje. |
| GeographicLib (vendorizado em `sdm/geocircle`) | `geographiclib-rs` (port em Rust) | Conferir os bits contra o vendorizado. Se não baterem, portar as funções usadas. |
| earcut (vendorizado em `io`) | `earcutr` | O mesmo algoritmo; conferir a ordem dos triângulos. |
| exactextract (vendorizado em `io`) | port | Não há crate equivalente fiel. |
| FFT do R (`nlm/rfft.cpp`) | port (não o `rustfft`) | A FFT do R tem outra ordem de operações. |
| GDAL (no Python, pelo rasterio; no Terraflux, carregado em tempo de execução) | nativo: continua o GDAL (`terraflux-gdal`, ou o crate `gdal`); WASM: `tiff` + o leitor de GeoTIFF do Terraflux | Não há GDAL em Rust puro. No navegador, só GeoTIFF/COG (e talvez PNG e ASCII grid). |
| PROJ (pyproj, GDAL) | nativo: o PROJ do GDAL; WASM: `proj4rs` | `proj4rs` cobre as projeções comuns, mas não as transformações por grade (NTv2, geoides) nem o banco EPSG completo do PROJ. As reprojeções no navegador podem diferir das nativas. |
| SIMD com despacho em tempo de execução (`_circuit_avx2`) | `#[target_feature(enable = "avx2")]` + `is_x86_feature_detected!` (ou o crate `multiversion`) | Um binário só, com as duas versões. No WASM, `simd128`. |

## Paridade

Quatro camadas, todas no CI:

1. **A suíte atual com o motor Rust.** `LANDSCHAFT_ENGINE=rust pytest` roda os mesmos testes contra as
   referências originais (R, Perl, GTB, Julia, Whitebox, Dinamica, etc.). Os testes não mudam, e as tolerâncias
   que já existem (ARM64, glibc, BLAS) continuam as mesmas.
2. **Testes diferenciais C++ × Rust** (`tests/parity/test_<módulo>.py`). Os mesmos casos dos
   `tests/*_cases.py`, mais cenas aleatórias com sementes fixas (tamanhos, NaN, NoData, bordas, uma linha e uma
   coluna, classes raras), passam pelos dois motores. A comparação é dos bits
   (`a.view(np.uint64) == b.view(np.uint64)`), para pegar `-0.0` e payloads de NaN, e cobre também as ordens de
   saída (listas de manchas, arestas, rotas).
3. **Independência do número de threads:** 1, 2, 3 e todas as threads dão os mesmos bits nos dois motores.
4. **Testes Rust (`cargo test`)** para as peças de base (`ls-math`, `ls-par`, heaps e sorts), contra valores
   gravados do C++ e com `proptest`. As partes `unsafe` passam pelo `miri`.

Plataformas: Windows x64 MSVC (a referência, onde as tabelas foram geradas), Linux x64, macOS arm64 (com as
tolerâncias de hoje) e `wasm32-unknown-unknown`. No WASM vale a mesma tolerância do ARM64 (poucas ULPs nas
funções elementares), porque lá não há x87 nem a mesma libm.

### Diferenças aceitas

Nenhuma até agora. Cada uma entra aqui com o kernel, a causa, o tamanho (em ULPs ou em células) e o teste que a
limita.

## Desempenho

- Os `tools/benchmark_*.py` ganham `--engine cpp|rust` e passam a medir os dois, alternados, com uma thread e
  com todas (como o `tools/benchmark.py` já faz).
- **Critério para apagar o C++ de um módulo:** a mediana do Rust fica em até 1,05× a do C++ em cada caso do
  benchmark, com 1 thread e com todas, no Windows e no Linux. Acima disso, otimiza-se antes de trocar: tirar
  checagens de limite nos laços quentes (iteradores, `chunks_exact`, `get_unchecked` revisado), SIMD explícito
  onde o C++ o tem, e o tamanho dos blocos do `ls-par`.
- Benchmarks finos com `criterion` nos crates, para pegar regressões sem passar pelo Python.
- O resultado de cada fase vai para `docs/benchmarks.md` (C++ × Rust).

## Fases

A ordem segue as dependências: as bases primeiro, depois os módulos pequenos e independentes (que validam o
caminho todo com pouco risco), depois os grandes e, por último, os de álgebra linear (o maior risco para a
paridade) e o MILP. Cada fase termina com release, commit e push.

0. **Infraestrutura e piloto.**
   - Workspace Cargo, `rust/ls-math`, `rust/ls-par`, `rust/ls-py`, Corrosion no CMake, `_rcore` no wheel,
     `landschaft._engine` e `LANDSCHAFT_ENGINE`.
   - CI: `cargo fmt`, `clippy`, `cargo test`, build `wasm32-unknown-unknown` dos crates, a suíte pytest com os
     dois motores e os testes de paridade. O cibuildwheel ganha o rustup nos contêineres manylinux e no macOS.
   - `ls-math`: x87 por `asm!`, `SoftExtended`, os geradores do R e da libstdc++, as somas e médias do R.
     Validado contra o `lsm/rmath.hpp` e o `lsm/extended.hpp`.
   - **Piloto: `serv`** (930 linhas, sem dependências). Portado, com paridade, benchmark, troca do padrão e o
     C++ apagado. No Terraflux, as funções do `serv` saem do shim.
1. **`hydro`, `rest`, `conn`.** Pequenos, com OpenMP simples: validam o `ls-par` em reduções e laços por linha.
2. **`gtb`** (MSPA, GTB/GuidosToolbox). Muito paralelo e muito medido (`tools/benchmark.py` contra o executável
   original): é a prova do desempenho.
3. **`lsm`** (métricas de paisagem). A base de vários outros; com ele portado, os crates seguintes não precisam
   mais do C++ do `lsm`.
4. **`nlm`** (modelos de paisagem neutra: geradores, FFT do R).
5. **`bio`** (Biodiverse: a matemática do Perl, aleatorizações).
6. **`io`** (warp do `ls.reproject`, zonal, exactextract, earcut, triangulação, reclassificação).
7. **`surf`, `sdm`, `sim`.** O `sdm` traz o GeographicLib; o `sim` traz os autovalores complexos do Eigen.
8. **`circuit`, `cost`** (Circuitscape, Omniscape, gdistance). A álgebra linear do Eigen: o Cholesky
   multifrontal, o `SparseLU` e as variantes AVX2. É a fase de maior risco para os bits.
9. **`prio`, `opt`** (priorização, MILP). O `opt` passa a usar o HiGHS pelo `highs-sys`, e os motores próprios
   (corte, fecho, cobertura, largo, zonas, `WatchScope`) são portados.
10. **`swat`** (SWAT+, cerca de 29 mil linhas, sem dependências nem OpenMP). Grande, mas mecânico.
11. **Encerramento.**
    - Sai o C++ do landschaft, o CMake (exceto o que o `highs-sys` usa), o pybind11 e o Eigen; o backend do wheel
      passa a ser o `maturin`.
    - No Terraflux saem o `terraflux-ls-sys` e o `cxx`.
    - O Terraflux compila para `wasm32-unknown-unknown`: o motor sem o `opt`, ou com o HiGHS para WASM (ver
      "Contras").

As fases 1 a 6 podem trocar de ordem se o Terraflux precisar de um módulo antes no WASM. As fases 0 e 8 não
mudam de lugar.

## Prós

- **Um toolchain só no Terraflux.** `cargo build`, sem CMake, sem o compilador C++ (exceto para o HiGHS), sem
  as 8,8 mil linhas do shim do `cxx` e sem os callbacks entre as linguagens. Builds mais rápidos e
  compilação cruzada mais simples.
- **WebAssembly de verdade.** `wasm32-unknown-unknown` com o motor e a interface egui no navegador, sem o
  Emscripten. É o objetivo da transição.
- **Um modelo de paralelismo só** (`rayon`) no landschaft e no Terraflux, sem dois pools de threads disputando
  os núcleos (o do OpenMP e o do `rayon`).
- **Wheels mais simples.** Sem runtime do OpenMP para empacotar (vcomp no Windows, libomp do conda-forge no
  macOS, gomp no Linux), sem baixar o Eigen no build e talvez com wheels `abi3` (um por plataforma).
- **Segurança de memória e de threads.** O compilador recusa corridas de dados e acessos fora dos limites (fora
  dos blocos `unsafe`), e somem classes inteiras de comportamento indefinido do C++ (estouro de inteiro com
  sinal, aliasing, iteradores invalidados).
- **Ponto flutuante explícito.** O Rust nunca contrai em FMA nem reordena somas, em nenhum compilador, então
  `/fp:precise` e `-ffp-contract=off` deixam de ser algo a lembrar.
- **Ferramentas melhores para testar:** `cargo test`, `proptest`, `criterion`, `miri` e `cargo-fuzz`, todos
  integrados.
- **O núcleo vira biblioteca Rust reutilizável:** outros projetos Rust podem usar os crates (crates.io), e não
  só o Python e o Terraflux.
- **O port serve de auditoria:** reler 100 mil linhas com testes diferenciais costuma achar bugs no código
  original.

## Contras e limitações

**Paridade bit a bit**
- O Eigen soma em blocos que dependem do tamanho do pacote SIMD e do bloqueio do GEMM. Bater com ele exige
  portar a mesma ordem de operações (Cholesky multifrontal, `SparseLU`, `EigenSolver`), e não dá para usar o
  `faer` ou o `nalgebra` nesses kernels. Se o port exato for caro demais em `circuit`, `cost` ou `sim`, a
  alternativa é aceitar poucas ULPs ali (decisão em aberto).
- `std::sort`, `nth_element` e `priority_queue` resolvem empates de um jeito que depende da STL. O
  `sort_unstable` e o `BinaryHeap` do Rust resolvem de outro. Onde isso muda o resultado, é preciso portar o
  algoritmo da STL.
- Pequenas diferenças semânticas a vigiar em cada kernel:
  - `f64::max` e `f64::min` ignoram o NaN, enquanto `std::max` devolve o primeiro argumento;
  - `as` de float para inteiro satura no Rust, enquanto no C++ é comportamento indefinido (o MSVC devolve
    `0x80000000`);
  - `powi` não é `std::pow(x, int)`;
  - o estouro de inteiro dá pânico em debug;
  - a formatação e a leitura de números em texto (`format!` e `printf("%g")`) diferem, o que pesa nos arquivos de
    texto do SWAT+ e nos formatos do Biodiverse.
- No WASM não há x87 e a libm é outra (musl), então os resultados no navegador ficam a poucas ULPs dos nativos,
  como no ARM64 hoje.

**Desempenho**
- O `rayon` tem mais overhead que o `schedule(static)` do OpenMP em laços curtos e finos: é preciso blocos
  maiores, e às vezes reorganizar o laço.
- As checagens de limite custam em laços quentes com índices calculados (vizinhanças, rotulagem de manchas),
  e tirá-las pede iteradores ou `unsafe` revisado.
- O LLVM e o MSVC vetorizam de modos diferentes: alguns kernels ficam mais rápidos, outros mais lentos, e cada um
  precisa ser medido.
- No navegador: SIMD só de 128 bits, memória limitada a 4 GB (wasm32), threads só com `SharedArrayBuffer` e
  COOP/COEP (o GitHub Pages não deixa pôr esses cabeçalhos), e o `wasm-bindgen-rayon` pede o Rust nightly com
  `build-std`. É de esperar 1,5× a 3× mais lento que o nativo.

**O que não tem equivalente em Rust**
- **HiGHS:** continua C++ (pelo `highs-sys`). O landschaft nunca fica 100% Rust na priorização. No WASM, o
  HiGHS precisa do Emscripten (o que traz de volta parte do problema) ou do `highs-js` num web worker, chamado
  pelo JavaScript. O patch (`highs_patch.cmake`) obriga a vendorizar uma cópia do HiGHS.
- **GDAL e PROJ:** no nativo continuam como estão (bindings para C). No navegador:
  - só GeoTIFF/COG e formatos simples, sem NetCDF, HDF, JPEG2000, GeoPackage, Shapefile, etc.;
  - reprojeção pelo `proj4rs`, sem transformações por grade nem o EPSG completo;
  - resultados de reprojeção diferentes dos nativos;
  - leitura e escrita de vetores bem mais limitadas.
- **Bibliotecas vendorizadas** (exactextract, GeographicLib, earcut): ou portes à mão, ou crates de terceiros
  cujos bits precisam ser conferidos.

**Custo e processo**
- São cerca de 100 mil linhas: muitos meses de trabalho.
- Durante a transição convivem dois motores. Um bug num módulo em port é corrigido duas vezes, e o wheel
  compila C++ e Rust (CI mais longo, build local mais pesado, rustup nos contêineres do cibuildwheel).
- A regra "o C++ para de crescer" atrasa recursos novos nos módulos ainda em C++, porque ou se porta o módulo
  antes, ou o recurso nasce em Rust e o C++ fica para trás.
- O `unsafe` continua necessário na fronteira com o Python (arrays do numpy), no `asm!` do x87 e em alguns
  laços quentes, e precisa de revisão e do `miri`.
- As mensagens de erro e os tipos de exceção precisam ser mapeados um a um para o Python não perceber a troca.
- O tamanho do binário pode mudar (monomorfização dos genéricos do Rust), assim como o tempo de importação do
  módulo.

## Decisões em aberto

- **Tolerância na álgebra linear.** Se o port exato do Eigen ficar caro demais (fase 8), aceitar poucas ULPs em
  `circuit`, `cost` e `sim`, ou insistir no bit a bit?
- **O C++ como referência depois de apagado.** Guardar uma tag (por exemplo `cpp-final`) e um job de CI que
  compare o Rust com ela de tempos em tempos, ou confiar só nas referências originais?
- **O MILP no navegador:** `highs-js` num worker, HiGHS pelo Emscripten, ou a priorização ficar fora do
  Terraflux web.

## Progresso

Nenhuma fase iniciada.
