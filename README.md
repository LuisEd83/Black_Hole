# 🕳️ Black Hole 

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

**Black Hole** é um simulador físico e visual de buracos negros desenvolvido em C++/Vulkan/CUDA. O objetivo é calcular com precisão como raios de luz se comportam próximos ao campo gravitacional intenso de um corpo compacto — e renderizar esse resultado em tempo real usando uma única API gráfica moderna.

O cálculo das geodésicas (trajetórias dos fótons no espaço-tempo curvo de Schwarzschild) é executado em um **compute shader Vulkan** com suporte a `shaderFloat64`. A composição final da cena ocorre via **Vulkan graphics** com dynamic rendering e um quad em tela cheia. Como objeto de estudo, usamos o buraco negro supermassivo da Via Láctea: **Sagittarius A\***.

---

## 🔭 Como Funciona

A simulação se baseia em três pilares físico-computacionais:

### 1. Ray Tracing por Pixel
Cada pixel da imagem corresponde a um raio lançado a partir de uma câmera virtual. O trajeto do raio é integrado numericamente, simulando como o fóton se propaga em um espaço-tempo distorcido pela gravidade.

### 2. Geodésicas em Espaço-Tempo Curvo (Compute Shader)
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

### 4. Pipeline Vulkan Compute → Graphics
O compute shader escreve os pixels **diretamente em uma imagem `VK_IMAGE_USAGE_STORAGE_BIT`**. Uma pipeline barrier transiciona a imagem para `SHADER_READ_ONLY_OPTIMAL`, e um fragment shader em tela cheia a amostra para exibição. Nenhuma transferência GPU→CPU é necessária para o framebuffer.

---

## ✨ Funcionalidades

- Integração RK4 de geodésicas nulas na métrica de Schwarzschild
- Paralelismo massivo por pixel via compute shader Vulkan (`shaderFloat64`)
- Renderização em tempo real via Vulkan graphics + GLFW com dynamic rendering
- Fundo estelar realista carregado de um starmap equirretangular
- Ruído de Perlin para textura turbulenta do disco de acreção
- Disco volumétrico com emissividade acumulada ao longo do raio
- Desvio Doppler relativístico e redshift gravitacional por pixel
- Passo adaptativo: step reduzido automaticamente próximo ao horizonte
- Early exit por parâmetro de impacto (raios que definitivamente escapam)
- Quad em tela cheia com `gl_VertexID`, sem VBO
- Double buffering (`MAX_FRAMES_IN_FLIGHT = 2`) para paralelismo CPU-GPU
- Suporte a redimensionamento de janela com recriação de swapchain
- Validation layers ativadas em builds de debug
- Build em C++20 estrito; warnings restantes são majoritariamente conversões de sinal e casts antigos, não erros
- Multiplataforma: **Linux**, **macOS** (MoltenVK) e **Windows**

---

## 📁 Estrutura do Projeto

```
Black_Hole/
├── include/                          # Headers C++ do projeto
│   ├── test_app.hpp                  # Classe orquestradora VulkanTestApp
│   ├── utils.hpp                     # Utilitários estáticos: readFile(), debugCallback()
│   ├── vk_init/
│   │   ├── setup.hpp                 # Setup + DeviceCapabilities + debug callback
│   │   ├── presentation.hpp          # Swapchain e image views
│   │   ├── compute_pipeline.hpp      # Pipeline de compute (kernel de ray tracing)
│   │   └── graphics_pipeline.hpp     # Pipeline gráfica (quad em tela cheia)
│   ├── vk_main/
│   │   └── render.hpp                # Render loop, descritores, sincronização e exportação
│   └── vk_utils/
│       └── vkimage.hpp               # Helpers de imagem Vulkan (criação, view, sampler, upload)
│
├── src/                              # Implementações C++
│   ├── test_app.cpp                  # VulkanTestApp (orquestrador)
│   ├── host.cpp                      # Bridge host para kernels CUDA (modo PNG)
│   ├── lodepng.cpp                   # Implementação da biblioteca lodepng
│   ├── perlin.cpp                    # Geração do ruído de Perlin
│   ├── starmap.cpp                   # Carregamento e amostragem do starmap
│   ├── temp_and_time.cpp             # Helpers de tempo e temperatura
│   ├── stb_image_write_impl.cpp      # Implementação STB image write
│   ├── vk_init/
│   │   ├── setup.cpp
│   │   ├── presentation.cpp
│   │   ├── compute_pipeline.cpp
│   │   └── graphics_pipeline.cpp
│   └── vk_main/
│       └── render.cpp
│
├── shaders/                          # Shaders Slang -> SPIR-V
│   ├── graphics.slang                # Vertex + Fragment (quad em tela cheia)
│   ├── compute.slang                 # Kernel de computação (ray tracing)
│   ├── constants.slang               # Constantes físicas
│   ├── geodesic.slang                # Integração geodésica
│   └── effects.slang                 # Efeitos visuais (Doppler, redshift)
│
├── cuda/                             # Kernels CUDA legados (modo PNG estático)
│   ├── raytrace.cu
│   ├── geod.cu
│   ├── effects.cu
│   ├── png_kernel.cu
│   └── internals/
│       └── *.cuh
│
├── headers/                          # Headers C++/CUDA compartilhados
│   ├── constants.hpp
│   ├── engine.hpp
│   ├── geodesic.cuh
│   ├── starmap.hpp
│   ├── perlin.hpp
│   └── ...
│
├── data/                             # Assets necessários em runtime
│   ├── starmap.png                   # Mapa estelar equirretangular
│   ├── starmap_2020_4k.exr           # Starmap alternativo em EXR
│   └── perlin.txt                    # Dados de ruído 3D pré-computados
│
├── icons/                            # Ícones da aplicação
├── output/                           # Imagens exportadas pelo simulador
├── video.cpp                         # Ponto de entrada do modo Vulkan (interativo / --export)
├── png.cpp                           # Ponto de entrada do modo CUDA estático
├── CMakeLists.txt                    # Build: modo video (Vulkan) ou png (CUDA)
├── CMakePresets.json                 # Presets de build por plataforma
├── requirements.sh                   # Script de instalação de dependências no Linux
├── .clangd                           # Configuração do clangd
├── .gitignore
└── LICENSE
```

---

## 🛠️ Pré-requisitos

| Dependência | Versão mínima | Finalidade |
|---|---|---|
| Vulkan SDK | 1.3+ | API de compute + graphics |
| CMake | 3.14+ | Build system |
| Compilador C++20 | GCC / Clang / MSVC | Código host |
| GLFW | 3.3+ | Janela e input |
| GLM | qualquer | Matemática 3D (tipos de vértice, matrizes MVP) |
| STB | header-only | Carregamento de imagens (`stb_image.h`) |
| Slang | qualquer | Compilação de shaders (`slangc`) |

### Por plataforma

<details>
<summary><b>Linux (Ubuntu/Debian/Fedora/Arch)</b></summary>

Você pode usar o script de instalação automática:

```bash
chmod +x requirements.sh
./requirements.sh
```

Ou instalar manualmente:

```bash
sudo apt install cmake ninja-build libglfw3-dev libglm-dev
# Vulkan SDK: https://vulkan.lunarg.com/sdk/home
# Slang: https://github.com/shader-slang/slang/releases
```

Compilador: GCC ou Clang com suporte a C++20.
</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install cmake ninja glfw glm
# Vulkan SDK (inclui MoltenVK): https://vulkan.lunarg.com/sdk/home
# Slang: https://github.com/shader-slang/slang/releases
```

> **Apple Silicon (M1/M2/M3):** suportado via MoltenVK. O compute shader usa `shaderFloat64`; verifique se seu dispositivo suporta isso através do MoltenVK.

Compilador: Clang (Xcode Command Line Tools).
</details>

<details>
<summary><b>Windows</b></summary>

1. Instale o [Visual Studio 2019 ou 2022](https://visualstudio.microsoft.com/) com o workload **"Desenvolvimento para Desktop com C++"**.
2. Instale o [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
3. Instale as dependências via **vcpkg**:

```powershell
vcpkg install glfw3 glm --triplet x64-windows
```

4. Configure o CMake com a integração do vcpkg:

```powershell
cmake --preset windows-release -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

Compilador: MSVC (recomendado) ou Clang-CL.
</details>

---

## 🚀 Build & Instalação

### Usando CMakePresets (recomendado)

Os presets usam o modo padrão `BUILD_MODE=video` (Vulkan). Para o modo CUDA estático, use a configuração manual abaixo.

```bash
# Linux
cmake --preset linux-release
cmake --build --preset linux-release

# macOS
cmake --preset macos-release
cmake --build --preset macos-release
```

```powershell
# Windows (PowerShell)
cmake --preset windows-release
cmake --build --preset windows-release
```

### Manual

```bash
git clone https://github.com/LuisEd83/Black_Hole.git
cd Black_Hole

# Modo Vulkan (renderização em tempo real / exportação estática)
cmake -B build -S . -G Ninja -DBUILD_MODE=video
cmake --build build

# Modo CUDA estático (legado, gera PNG via OpenGL/CUDA)
cmake -B build -S . -G Ninja -DBUILD_MODE=png
cmake --build build
```

> O executável gerado fica em `./build/black_hole_sim`.
> Os shaders são compilados automaticamente pelo CMake: `slangc` traduz `shaders/graphics.slang` para `shaders/graphics.spv` e `shaders/compute.slang` para `shaders/compute.spv`.

---

## 🎮 Uso

### Modo interativo (padrão)

Execute o binário sem argumentos. Uma janela GLFW abrirá exibindo a simulação em tempo real:

```bash
# Linux / macOS
./build/black_hole_sim

# Windows
build\black_hole_sim.exe
```

**Controles do teclado:**

| Entrada | Ação |
|---|---|
| `W` | Mover para frente |
| `S` | Mover para trás |
| `A` | Mover para a esquerda |
| `D` | Mover para a direita |
| `Q` | Mover para baixo |
| `E` | Mover para cima |
| `←` / `→` | Rotacionar câmera (yaw) |
| `↑` / `↓` | Rotacionar câmera (pitch) |
| `Esc` | Fechar janela |

### Modo de exportação PNG

Execute com a flag `--export` para renderizar um único frame e salvá-lo como PNG (sem janela):

```bash
# Exporta com padrão: output.png em 1920×1080
./build/black_hole_sim --export

# Exporta com nome customizado
./build/black_hole_sim --export minha_imagem.png

# Exporta com nome e resolução customizados
./build/black_hole_sim --export minha_imagem.png 2560 1440
```

> ⚠️ Para definir uma resolução customizada, você deve também fornecer o nome do arquivo. A resolução padrão é **1920×1080**.

Exemplo verificado:

```bash
./build/black_hole_sim --export test_output.png 800 600
```

### Arquivos de dados necessários

Os seguintes arquivos devem estar presentes em `data/` (relativo ao executável):

```
data/
├── starmap.png   # Mapa estelar equirretangular (ex: ESA Gaia DR2)
└── perlin.txt    # Dados de ruído de Perlin 3D pré-computados
```

---

## ⚙️ Configuração

### Parâmetros físicos (shaders)

As constantes físicas (raio de Schwarzschild, fatores de passo, etc.) são definidas diretamente nos shaders compute (`shaders/constants.slang`):

```slang
static const double MAX_STEPS        5000    // iterações máximas por raio
static const double STEP_FACTOR      0.5     // tamanho do passo em unidades de rs
static const double IMPACT_CUTOFF    7.5     // threshold de escape antecipado
static const double ADAPTIVE_FACTOR  5.0     // raio (em rs) onde o passo diminui
static const double EMISSIVITY_RATE  0.001   // limiar mínimo de emissividade do disco
```

### Parâmetros da câmera

A câmera é inicializada com valores fixos em `src/vk_main/render.cpp` (posição em unidades de `rs`):

```cpp
// Posição inicial (distância = 12.0 rs, com offsets x=1.0, y=1.1, z=0.7)
cameraPos = glm::dvec3(12.0, 13.2, 8.4);

// FOV vertical padrão
fov_y = 60.0f;
```

Para ajustar a posição inicial no modo interativo, modifique `Render::processKeyboardInput()` em `src/vk_main/render.cpp`. Para ajustar no modo de exportação, modifique `Render::exportToImage()` no mesmo arquivo.

### Resolução de exportação

No modo interativo, a resolução segue a janela (800×600 por padrão, redimensionável). No modo de exportação, a resolução é configurada por argumentos de linha de comando (veja [Uso](#-uso) acima).

---

## 🗺️ Roadmap

- [x] Arquitetura modular do app Vulkan (hierarquia de subclasses)
- [x] Vulkan-Hpp RAII com dynamic rendering
- [x] Compilação Slang → SPIR-V
- [x] Mapeamento de texturas com staging buffers
- [x] Uniform buffers com transforms MVP
- [x] Descriptor sets para UBOs e combined image samplers
- [x] Swapchain duplo com sincronização adequada
- [x] Suporte a redimensionamento de janela
- [x] Compute pipeline com `shaderFloat64`
- [x] Storage image para saída do compute
- [x] Compute descriptor set (storage image, UBO, starmap, perlin)
- [x] Portar `geod.cu`, `effects.cu`, `raytrace.cu` para Slang/GLSL com `dvec3`
- [x] Display de quad em tela cheia com saída do compute
- [x] Gravação de command buffers: compute dispatch → barrier → graphics render
- [x] Controles de câmera (teclado WASD + setas)
- [x] Modo de exportação PNG
- [ ] Órbita esférica da câmera com mouse (drag, scroll zoom)
- [ ] Interface interativa para ajuste de parâmetros em tempo real
- [ ] Exportação de vídeo frame-a-frame
- [ ] Empacotamento Docker para build reprodutível

---

## 📚 Referências

- [James, O. et al. (2015) — *Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar*](https://iopscience.iop.org/article/10.1088/0264-9381/32/6/065001)
- [Luminet, J.-P. (1979) — *Image of a spherical black hole with thin accretion disk*](https://www.aanda.org/articles/aa/full_html/2019/01/aa14506-19/aa14506-19.html)
- [Misner, Thorne & Wheeler — *Gravitation* (1973)](https://press.princeton.edu/books/hardcover/9780691177793/gravitation)
- [Vulkan Specification — Khronos Group](https://www.vulkan.org/)
- [LearnOpenGL](https://learnopengl.com/)

---

## 📄 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo [`LICENSE`](./LICENSE) para mais detalhes.
