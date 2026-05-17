# 🕳️ Black Hole Sim

> Renderização de buracos negros usando **Ray Tracing**, **OpenGL** e aceleração massiva de GPU via **CUDA**.  
> Projeto acadêmico com foco em renderização física e paralelismo em GPU.

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Como Funciona](#-como-funciona)
- [Funcionalidades](#-funcionalidades)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Build & Instalação](#-build--instalação)
- [Uso](#-uso)
- [Configuração](#-configuração)
- [Roadmap](#-roadmap)
- [Referências](#-referências)
- [Licença](#-licença)

---

## 🌌 Sobre o Projeto

**Black Hole Sim** é um simulador físico e visual de buraco negro desenvolvido em C++/CUDA. O objetivo é calcular com precisão como raios de luz se comportam próximos ao campo gravitacional intenso de um corpo compacto — e renderizar esse resultado em tempo real na tela.

O cálculo das geodésicas (trajetórias dos fótons no espaço-tempo curvo de Schwarzschild) é executado em paralelo na GPU via **CUDA**. A composição final da cena ocorre via **OpenGL** com shaders GLSL. Como objeto de estudo, usamos o buraco negro supermassivo da Via Láctea: **Sagittarius A\***.

---

## 🔭 Como Funciona

A simulação se baseia em três pilares físico-computacionais:

### 1. Ray Tracing por Pixel
Cada pixel da imagem corresponde a um raio lançado a partir de uma câmera virtual. O trajeto do raio é integrado numericamente, simulando como o fóton se propaga em um espaço-tempo distorcido pela gravidade.

### 2. Geodésicas em Espaço-Tempo Curvo (CUDA)
As equações diferenciais das geodésicas nulas (métrica de Schwarzschild) são integradas com um integrador **Runge-Kutta de 4ª ordem** (RK4). Para cada raio, o kernel determina um de quatro resultados possíveis:

| Resultado | Descrição |
|---|---|
| `HORIZON` | Raio capturado pelo horizonte de eventos → pixel preto |
| `ESCAPE` | Raio escapa e atinge o fundo estelar → amostra do starmap |
| `DISK` | Raio atravessa o disco de acreção → composição volumétrica |
| `FALLBACK` | Raio orbita indefinidamente → tratamento por distância |

### 3. Efeitos Físicos do Disco de Acreção
A aparência do disco incorpora três efeitos relativísticos:

- **Desvio Doppler** — O lado do disco que se aproxima da câmera emite luz comprimida (azulada e mais brilhante); o lado que se afasta emite luz esticada (avermelhada e mais escura).
- **Redshift Gravitacional** — Fótons emitidos mais próximos ao horizonte perdem energia ao escapar do campo gravitacional intenso, tornando as regiões internas mais vermelhas.
- **Disco Volumétrico** — Em vez de detecção por cruzamento de plano (disco infinitamente fino), a emissividade é acumulada ao longo do caminho do raio dentro do volume do disco, produzindo bordas suaves e múltiplas camadas sobrepostas.

### 4. Renderização OpenGL (CUDA-GL Interop)
No modo interativo, o kernel CUDA escreve os pixels **diretamente na textura OpenGL** via `cudaSurfaceObject_t`, evitando qualquer transferência GPU→CPU. A textura é então renderizada por um quad em tela cheia com shaders GLSL simples.

---

## ✨ Funcionalidades

- Integração RK4 de geodésicas nulas na métrica de Schwarzschild
- Paralelismo massivo por pixel via CUDA (Persistent Threads no modo interativo)
- Renderização em tempo real via OpenGL + GLFW com CUDA-GL Interop
- Fundo estelar realista carregado de um starmap equirretangular
- Ruído de Perlin para textura turbulenta do disco de acreção
- Disco volumétrico com emissividade acumulada ao longo do raio
- Desvio Doppler relativístico e redshift gravitacional por pixel
- Passo adaptativo: step reduzido automaticamente próximo ao horizonte
- Early exit por parâmetro de impacto (raios que definitivamente escapam)
- Monitoramento em tempo real de temperatura de GPU/CPU no terminal
- Dois modos de saída: janela OpenGL interativa ou exportação PNG
- Suporte a múltiplas resoluções (800×600 até 4096×2048)
- Multiplataforma: **Linux**, **macOS** (Intel) e **Windows**

---

## 📁 Estrutura do Projeto

```
Black_Hole/
├── cuda/
│   ├── geodesic.cu        # Kernel principal: integração RK4 + composição de cor
│   ├── geodesic.cuh       # Header do kernel (struct Rays, constantes físicas)
│   ├── comms.cu           # activateSetFlags, getStateCountsPtr
│   ├── comms.cuh          # Declarações de comms
│   └── feedbacks.cu       # Estimativa de tempo e warmup do driver CUDA
├── shaders/
│   ├── display.vert       # Quad em tela cheia (procedural, sem VBO)
│   └── display.frag       # Amostragem da textura CUDA→OpenGL
├── src/
│   ├── engine.cpp/.hpp    # Loop principal OpenGL, CUDA-GL Interop, callbacks de câmera
│   ├── host.cpp           # Ponte CPU→GPU: converte vetores GLM → double3, chama launchRaytrace
│   ├── constants.hpp      # Parâmetros globais: resolução, steps, fatores físicos
│   ├── distribution.hpp   # StateHeatmap: display de progresso em tempo real (std::thread)
│   ├── starmap.cpp/.hpp   # Carregamento e textura CUDA do mapa estelar
│   ├── perlin.cpp/.hpp    # Carregamento e textura CUDA do ruído de Perlin
│   ├── temp_and_time.cpp  # Leitura de temperatura GPU/CPU, helpers de tempo
│   └── lodepng.cpp/.h     # Codificação PNG (LodePNG)
├── data/
│   ├── starmap.png        # Mapa estelar equirretangular (não incluído no repo)
│   └── perlin.txt         # Dados de ruído pré-computados (não incluído no repo)
├── main.cpp               # Modo interativo: configura câmera e chama engineRun()
├── png.cpp                # Modo exportação: renderiza e salva PNG via LodePNG
├── CMakeLists.txt         # Build system (CMake 3.24+)
├── CMakePresets.json      # Presets para Linux, macOS e Windows
└── LICENSE
```

---

## 🛠️ Pré-requisitos

É necessária uma **GPU NVIDIA** com suporte a CUDA. A arquitetura é detectada automaticamente via `CUDA_ARCHITECTURES native`.

### Dependências comuns

| Dependência | Versão mínima | Finalidade |
|---|---|---|
| CUDA Toolkit | 11.0+ | Compilação dos kernels GPU |
| CMake | 3.24+ | Build system |
| OpenGL | 3.3 Core | Renderização |
| GLEW | qualquer | Carregamento de extensões OpenGL |
| GLFW3 | 3.4+ | Janela e contexto OpenGL |
| GLM | qualquer | Matemática 3D |

### Por plataforma

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
sudo apt install cmake ninja-build libglfw3-dev libglew-dev libglm-dev
# CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
```

Compilador: GCC ou Clang com suporte a C++17.
</details>

<details>
<summary><b>macOS (Intel)</b></summary>

```bash
brew install cmake ninja glfw glew glm
# CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
```

> **Apple Silicon (M1/M2/M3):** não suportado. A NVIDIA encerrou o suporte ao CUDA em macOS a partir do macOS Mojave para placas externas, e o Apple Silicon não possui suporte a CUDA de forma alguma.

Compilador: Clang (Xcode Command Line Tools).
</details>

<details>
<summary><b>Windows</b></summary>

1. Instale o [Visual Studio 2019 ou 2022](https://visualstudio.microsoft.com/) com o workload **"Desenvolvimento para Desktop com C++"**
2. Instale o [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. Instale as dependências via **vcpkg**:

```powershell
vcpkg install glfw3 glew glm --triplet x64-windows
```

4. Configure o CMake com a integração do vcpkg:

```powershell
cmake --preset windows-vs2022 -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

Compilador: MSVC (obrigatório para CUDA no Windows — MinGW não é suportado pelo NVCC).
</details>

---

## 🚀 Build & Instalação

### Usando CMakePresets (recomendado)

```bash
# Linux
cmake --preset linux-release
cmake --build --preset linux-release

# macOS (Intel)
cmake --preset macos-release
cmake --build --preset macos-release
```

```powershell
# Windows (PowerShell)
cmake --preset windows-vs2022
cmake --build --preset windows-vs2022
```

### Manual

```bash
git clone https://github.com/LuisEd83/Black_Hole.git
cd Black_Hole

cmake -B build -S .
cmake --build build --config Release
```

> Os shaders e os arquivos de `data/` são copiados automaticamente para o diretório do executável pelo CMake após o build.

---

## 🎮 Uso

### Modo interativo (padrão)

Execute o binário gerado. Uma janela OpenGL abrirá exibindo a simulação em tempo real:

```bash
./build/BlackHoleCUDA       # Linux / macOS
build\Release\BlackHoleCUDA.exe  # Windows
```

**Controles:**

| Entrada | Ação |
|---|---|
| Arrastar mouse (botão esquerdo) | Orbitar câmera |
| Scroll do mouse | Zoom (ajusta raio orbital) |
| `Esc` | Fechar |

### Arquivos de dados necessários

Os seguintes arquivos devem estar presentes em `data/` (relativo ao executável):

```
data/
├── starmap.png   # Mapa estelar equirretangular (ex: ESA Gaia DR2)
└── perlin.txt    # Dados de ruído de Perlin pré-computados
```

---

## ⚙️ Configuração

As principais variáveis de controle estão em `src/constants.hpp`:

### Modo de saída

```cpp
const bool is_gl = true;   // true  → janela OpenGL interativa (main.cpp)
                           // false → exportação PNG (png.cpp)
```

> Para usar `is_gl = false`, é necessário compilar `png.cpp` como ponto de entrada no lugar de `main.cpp`.

### Resolução

```cpp
const std::string res = "Minimal";  // define WIDTH e HEIGHT
```

| Valor | Resolução |
|---|---|
| `"Minimal"` | 800 × 600 |
| `"HD"` | 1280 × 720 |
| `"HD+"` | 1600 × 900 |
| `"FHD"` | 1920 × 1080 |
| `"QHD"` | 2560 × 1440 |
| `"UHD"` | 3840 × 2160 |
| `"4K"` | 4096 × 2048 |

### Parâmetros físicos

```cpp
constexpr int    MAX_STEPS      = 5000;   // iterações máximas por raio
constexpr double STEP_FACTOR    = 0.5;    // tamanho do passo em unidades de rs
constexpr double IMPACT_CUTOFF  = 7.5;    // threshold de escape antecipado
constexpr double ADAPTIVE_FACTOR = 5.0;  // raio (em rs) em que o step começa a diminuir
constexpr double EMISSIVITY_RATE = 0.001; // limiar mínimo de emissividade do disco
```

### Câmera (`main.cpp`)

```cpp
const double cam_dist      = RS * BH::factor;  // distância da câmera ao buraco negro
float fov_y                = 60.0f;            // campo de visão vertical (graus)
glm::vec3 target           = glm::vec3(...);   // ponto para onde a câmera aponta
```

---

## 🗺️ Roadmap

- [x] Integração RK4 de geodésicas nulas (Schwarzschild)
- [x] Disco de acreção volumétrico com emissividade acumulada
- [x] Desvio Doppler relativístico e redshift gravitacional
- [x] CUDA-GL Interop (zero cópia GPU→CPU no modo interativo)
- [x] Passo adaptativo próximo ao horizonte de eventos
- [x] Monitoramento de temperatura GPU/CPU em tempo real
- [x] Suporte multiplataforma: Linux, macOS, Windows
- [ ] Suporte à métrica de Kerr (buraco negro em rotação)
- [ ] Interface interativa para ajuste de parâmetros em tempo real
- [ ] Exportação de vídeo frame-a-frame
- [ ] Empacotamento Docker para build reprodutível

---

## 📚 Referências

- [James, O. et al. (2015) — *Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar*](https://iopscience.iop.org/article/10.1088/0264-9381/32/6/065001)
- [Luminet, J.-P. (1979) — *Image of a spherical black hole with thin accretion disk*](https://www.aanda.org/articles/aa/full_html/2019/01/aa14506-19/aa14506-19.html)
- [Misner, Thorne & Wheeler — *Gravitation* (1973)](https://press.princeton.edu/books/hardcover/9780691177793/gravitation)
- [CUDA C++ Programming Guide — NVIDIA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [LearnOpenGL](https://learnopengl.com/)

---

## 📄 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo [`LICENSE`](./LICENSE) para mais detalhes.
