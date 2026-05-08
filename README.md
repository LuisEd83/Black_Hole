# black_hole — renderizando buracos-negros em OpenGL com Aceleração de GPU via CUDA.

## Overview

Esse documento cobre tanto a parte programática quanto a parte conceitual sobre a concepção
de simulações de buracos-negros em C++/CUDA. O projeto é puramente acadêmico, com perspectivas
de aprendizado e captação de conhecimento sobre renderização de imagem e aceleração de GPU, 
bem como a satisfação de simular um corpo tão grandioso! 

# 🕳️ Black Hole Sim
 
> Renderização em tempo real de buracos-negros usando **Ray Tracing**, **OpenGL** e aceleração de GPU via **CUDA**.

<!-- No caso das imagens, deve-se arrastar o arquivo de imagem diretamente para o editor do README ou usar a sintaxe Markdown: 
![Alt text](caminho/para/imagem.png). -->

![image1]

![image2]

![image3]

![image4]

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

 
## 🌌 Sobre o Projeto

**Black Hole Sim** é um projeto puramente acadêmico que visa explorar a simulação física e visual de um buraco negro a partir de métodos computacionais. A proposta é calular como os raios de luz se comportam próximos ao campo gravitacional gerado pelo corpo compacto, isto é, um corpo de massa extrema compactado em um pequeno volume. 

O cálculo das geodésicas (trajetórias de raios de luz no espaço-tempo curvo) é realizado em paralelo na GPU via **CUDA**, enquanto a renderização final da cena ocorre via **OpenGL** com shaders GLSL personalizados.

Ademais, para a simulação utilizamos como exemplo o buraco negro supermassivo da nossa Via Láctea: Sagittarius A*.
 
## 🔭 Como Funciona
 
A simulação se baseia em três pilares físico-computacionais:

**1. Ray Tracing**
Cada pixel da imagem corresponde a um raio lançado a partir de uma câmera virtual. O trajeto do raio é computado usando métodos numéricos, simulando como o fóton se comporta em um espaço-tempo distorcido por um buraco negro.

**2. Geodésicas em Espaço-Tempo Curvo (CUDA)**
As equações diferenciais das geodésicas nulas (raios de luz) são integradas numericamente na GPU. Isso permite calcular, para cada raio, se ele escapa do buraco-negro, cai no horizonte de eventos, ou é redirecionado para alguma região do fundo estrelado.

**3. Renderização OpenGL**
O resultado é composto com um fundo estelar real (`starmap.png`), ruído procedural de Perlin e shaders GLSL que aplicam efeitos visuais como o disco de acreção, o desvio para o vermelho (redshift) gravitacional e o desvio para o azul (blueshift) gravitacional.

## ✨ Funcionalidades

- Simulação física de trajetórias de luz sob lente gravitacional intensa
- Aceleração massiva via CUDA (paralelismo por pixel)
- Renderização em tempo real com OpenGL + GLFW
- Fundo estelar realista carregado de um starmap real
- Ruído de Perlin para textura do disco de acreção
- Suporte a múltiplas resoluções de saída
- Exportação de frames em PNG via LodePNG

## 📁 Estrutura do Projeto
 
```
Black_Hole/
├── cuda/
│   ├── geodesic.cu        # Integração numérica das geodésicas na GPU
│   ├── comms.cu           # Comunicação host ↔ device
│   └── feedbacks.cu       # Feedback de parâmetros físicos
├── shaders/
│   ├── *.vert             # Vertex shaders GLSL
│   └── *.frag             # Fragment shaders GLSL
├── src/
│   ├── engine.cpp/.hpp    # Loop principal de renderização OpenGL
│   ├── host.cpp           # Interface CPU com os kernels CUDA
│   ├── constants.hpp      # Constantes físicas (Schwarzschild, etc.)
│   ├── starmap.cpp/.hpp   # Carregamento e consulta do mapa estelar
│   ├── perlin.cpp/.hpp    # Geração de ruído de Perlin
│   ├── temp_and_time.cpp  # Temperatura e timing da simulação
│   └── lodepng.cpp        # Codificação/decodificação PNG
├── main.cpp               # Ponto de entrada: configuração de câmera e engine
├── png.cpp                # Utilitários auxiliares de imagem
├── CMakeLists.txt
└── LICENSE
```

## 🛠️ Pré-requisitos
 
Certifique-se de ter instalado:
 
| Dependência | Versão mínima | Finalidade |
|---|---|---|
| GCC / Clang | C++17 | Compilação do código host |
| NVCC (CUDA Toolkit) | 11.0+ | Compilação dos kernels GPU |
| CMake | 3.24+ | Sistema de build |
| OpenGL | 3.3+ | Renderização |
| GLEW | qualquer | Extensões OpenGL |
| GLFW3 | 3.x | Janela e contexto OpenGL |
| GLM | qualquer | Matemática 3D (vetores/matrizes) |
 
> **Nota:** É necessária uma GPU NVIDIA com suporte a CUDA. A arquitetura é detectada automaticamente via `CUDA_ARCHITECTURES native`.

## 🚀 Build & Instalação
 
```bash
# 1. Clone o repositório
git clone https://github.com/LuisEd83/Black_Hole.git
cd Black_Hole
 
# 2. Crie o diretório de build
mkdir build && cd build
 
# 3. Configure com CMake
cmake ..
 
# 4. Compile
cmake --build . --config Release
 
# 5. Execute (a partir da raiz do projeto ou do diretório de saída)
./BlackHoleCUDA
```
 
> Os shaders são copiados automaticamente para o diretório de saída pelo CMake após o build.
 
 ## 🎮 Uso

Ao executar o binário, uma janela OpenGL abrirá exibindo a simulação em tempo real. A câmera é posicionada automaticamente a partir dos parâmetros definidos em `main.cpp`.

Os dados de suporte necessários devem estar presentes em:
 
```
data/
├── starmap.png   #Mapa estelar de fundo
└── perlin.txt    #Dados de ruído pré-computados
```

## ⚙️ Configuração

As principais variáveis de controle estão localizadas em `src/constants.hpp` e no início de `main.cpp`:

**Renderização** - Altera entre a simulação em tempo real e a renderização de uma imagem estática do tipo .png. Elas podem ser alterados diretamente em `src/constants.hpp`
```hpp
const bool is_gl = false; //Neste caso está habilitado a renderização de imagem estática
```


**Resolução de saída** — definida pela constante `BH::res`:
 
| Valor | Resolução |
|---|---|
| `"Minimal"` | 800 × 600 |
| `"HD"` | 1280 × 720 |
| `"HD+"` | 1600 × 900 |
| `"FHD"` | 1920 × 1080 |
| `"QHD"` | 2560 × 1440 |
| `"UHD"` | 3840 × 2160 |
| `"4K"` | 4096 × 2048 |

**Câmera** — os parâmetros de posição, alvo e campo de visão podem ser ajustados diretamente em `main.cpp`:
 
```cpp
const double graus          = 10.0;   // azimute da câmera
const double elevation_angle = 5.0;   // ângulo de elevação
float fov_y                 = 60.0f;  // campo de visão vertical
```
 
## 🗺️ Roadmap
 
- [ ] Interface interativa para ajuste de parâmetros em tempo real
- [ ] Suporte a rotação do buraco-negro (métrica de Kerr)
- [ ] Exportação de vídeo frame-a-frame
- [ ] Desvio para o vermelho (redshift) gravitacional nos pixels do disco
- [ ] Multithreading CPU para pré-processamento de dados
- [ ] Empacotamento Docker para build reprodutível

## 📚 Referências
 
- [James, O. et al. (2015) — *Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar*](https://iopscience.iop.org/article/10.1088/0264-9381/32/6/065001)
- [Luminet, J.-P. (1979) — *Image of a spherical black hole with thin accretion disk*](https://www.aanda.org/articles/aa/full_html/2019/01/aa14506-19/aa14506-19.html)
- [Misner, Thorne & Wheeler — *Gravitation* (1973)](https://press.princeton.edu/books/hardcover/9780691177793/gravitation)
- [CUDA C++ Programming Guide — NVIDIA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [LearnOpenGL — Ray Tracing concepts](https://learnopengl.com/)
---
 
## 📄 Licença
 
Distribuído sob a licença **MIT**. Consulte o arquivo [`LICENSE`](./LICENSE) para mais detalhes.
 