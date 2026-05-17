#include "engine.hpp"
#include "constants.hpp"
 
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
 
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

#include "platform.hpp"
#if BH_CPU_BACKEND
#include <thread>
#include <atomic>
#include "cpu_raytrace.hpp"
#include "starmap.hpp"
#include "perlin.hpp"
static const double rs_engine = 2.0 * G * BH_MASS / (c * c);
#endif

using namespace std;
using namespace glm;


/*

    mini-resumo de OpenGL:
    
    •   GLFW: faz o processo de background, criação de janelas, contexto na OS de
        que há uma janela OpenGL rodando, i/o mouse e teclado.
    
    •   GLEW: carrega todas as funções primitivas da API OpenGL 1.1.
    
    •   glm/glm.hpp & glm/gtc/matrix_transform.hpp: headers matemáticos que dão
        ao programador a liberdade de escrever com tipos da própria OpenGL (vec3, mat4...).
        Nesse contexto, é usada para transformação de matriz em relação ao movimento da câmera.
    
    •   OpenGL: uma API que se relaciona com o driver de qualquer GPU.
        
        → Trabalha com a criação de triângulos para renderização de objetos.
        → Esses triângulos tem vértices, em que cada triângulo terá operações computadas para esses vértices.
        → Esses vértices tem parâmetros regidos por um VAO (Vertex Array Object). Quando não há atributos, ele é nulo.   
        → A pipeline desse programa será: dados → CUDA Kernel → OpenGL → janela.
        


*/

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// declaração evitar dependência circular de .hpp

void raytraceCUDA(bool is_sim, void* pixels,
                  int WIDTH, 
                  int HEIGHT,
                  vec3 pos, vec3 fwd, vec3 right, vec3 up,
                  float fov_y
                  );
 


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#if !BH_CPU_BACKEND
static cudaGraphicsResource_t cuda_tex_resource = nullptr;
#endif

static string loadShaders(const string& path){
    
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "[ENGINE]: Não foi possível abrir shader: " << path << "\n";

        return "";
    }

    ostringstream buf;
    buf << file.rdbuf();

    return buf.str();
}



static GLuint compileShaders(GLenum type, const char* src){

    /*    
        helper para compilar e resgatar informação do processamento de shaders.
    */


    GLuint id = glCreateShader(type);
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id); 

    GLint ok;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if(!ok){
        char log[512];
        glGetShaderInfoLog(id, 512, nullptr, log);
    
        cerr << "[ENGINE]: Erro ao compilar shaders: \n" << log << "\n";
    }
    
    return id;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


static GLuint buildProgram(const char* vert, const char* frag){
    
    /* 
        
        aqui o porgrama é montado: com vert e frag, são calculados os valores
        de cada vértice nos triângulos, e então são conectados ao projeto.

        esse processo ocorre uma vez, como se fosse um mapa. o loop apenas repete o processamento
        das variáveis e o movimento das funções.

    */

    GLuint vertex   = compileShaders(GL_VERTEX_SHADER, vert);
    GLuint fragments= compileShaders(GL_FRAGMENT_SHADER, frag);
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragments);
    glLinkProgram(program);
     

    GLint ok;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        
        cerr << "[ENGINE]: Erro ao linkar programa:\n" << log << "\n";
    
    }

    glDeleteShader(vertex);
    glDeleteShader(fragments);

    return program;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



static GLuint makeGLFrameTexture(int WIDTH, int HEIGHT){
        
    /*
        função que prepara uma textura vazia na GPU que será eventualmente completa pelos
        pixels CUDA.
            
            → glGenTextures reserva um id,
            → glBindTexture binda, leva a textura vazia à textura na GPU,
            → glTexImage2D aloca memória,
            → os Parametri definem não intepolação, executar CLAMP em [0,1],
            → glBindTexture(0) desbinda para evitar chamadas acidentais.
    */


    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    
    return texture;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


static CameraState camera;

static void mouseButtons(GLFWwindow*, int button, int action, int){

    /*
        processando eventual click do mouse: filtra drag como o press do botão esquerdo.   
    */

    if(button == GLFW_MOUSE_BUTTON_LEFT){ camera.dragging = (action == GLFW_PRESS); }
}

static void cursorPos(GLFWwindow* window, double x, double y){
    
    /*
    
    */

    const float sensitivity = 10e-10f;

    if(!camera.dragging){
        
        camera.last_mouse_x = x;
        camera.last_mouse_y = y;

        return;
    }

    float dx = float(x - camera.last_mouse_x);
    float dy = float(y - camera.last_mouse_y);

    camera.last_mouse_x = x;
    camera.last_mouse_y = y;
    

    camera.azimuth += dx * sensitivity;
    camera.elevation += dy * sensitivity;
    
    // apenas uma garantia para a câmera não poder virar de cabeça pra baixo
    const float limit = radians(89.0f);
    if(camera.elevation > limit) camera.elevation = limit;
    if(camera.elevation < -limit) camera.elevation = -limit;
    

    camera.will_rerender = true;
}


static void scrollEvent(GLFWwindow*, double, double y_off){
    
    camera.orbital_radius -= float(y_off) * camera.orbital_radius * 0.1f;
    if(camera.orbital_radius < 1.0f) camera.orbital_radius = 1.0f;


    camera.will_rerender = true;
}


static void keypressEvent(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/){

    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);

}


static void frameBuffer(GLFWwindow*, int WIDTH, int HEIGHT){
    
    /*
        fallback para redimensionamento de janela.
        o correto seria redimensionar via: 
            fetch OS WIDTH, HEIGHT
                ↓
            passar pro programa
                ↓
            passar pro kernel

        alternativa: só deixar a janelinha quietinha.
    */

    glViewport(0,0, WIDTH, HEIGHT);
}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



void engineRun(const EngineConfig& config){

    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 1: iniciar GLFW
    if(!glfwInit()){
        cerr << "[ENGINE]: Falha ao iniciar GLFW.\n";
        return;
    }
    
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    GLFWwindow* window = glfwCreateWindow(config.WIDTH, config.HEIGHT, config.title.c_str(), nullptr, nullptr);
    
    if(!window){
        cerr << "[ENGINE]: Falha ao criar janela GLFW.\n";         
        glfwTerminate();

        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); 

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 2: iniciar GLEW
    

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){
        cerr << "[ENGINE]: Falha ao iniciar GLEW.\n";
        
        glfwDestroyWindow(window);
        glfwTerminate();

        return;
    }

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 3: coordenação de IO (callbacks)
        

    glfwSetMouseButtonCallback(window, mouseButtons);
    glfwSetCursorPosCallback(window, cursorPos);
    glfwSetScrollCallback(window, scrollEvent);
    glfwSetKeyCallback(window, keypressEvent);
    glfwSetFramebufferSizeCallback(window, frameBuffer);


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 4: settar estado inicial da camera


    camera.target   = config.cam_target;
    camera.world_up = config.cam_up;
    camera.orbital_radius = length(config.cam_pos - config.cam_target);
    
    vec3 dir = normalize(config.cam_pos - config.cam_target);
    camera.elevation = asin(dir.z);
    camera.azimuth = atan2(dir.y, dir.x);
    camera.will_rerender = true;


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 5: recursos do OpenGL

        
    string vert = loadShaders("shaders/display.vert");
    string frag = loadShaders("shaders/display.frag");
    const char* vert_str = vert.c_str();
    const char* frag_str = frag.c_str();
    
    GLuint program  = buildProgram(vert_str, frag_str);
    GLuint textures = makeGLFrameTexture(config.WIDTH,  config.HEIGHT);
    
#if !BH_CPU_BACKEND
    cudaGraphicsGLRegisterImage(&cuda_tex_resource, textures,
                             GL_TEXTURE_2D,
                             cudaGraphicsRegisterFlagsSurfaceLoadStore);
#endif

    // criaçao do VAO vazio
    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_frame"), 0); 
    // textura unit 0. funciona como um índice, você pode armazenar várias texturas.
    // nesse caso, temos apenas uma, logo índice = 0. 
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 6: reservar buffer de pixels para __host__


    vector<unsigned char>pixels;
    pixels.resize(config.WIDTH * config.HEIGHT * 3); // 3 canais de RGB

#if BH_CPU_BACKEND
    std::thread        cpu_thread;
    std::atomic<bool>  cpu_done{true};
    bool               cpu_upload_pending = false;
    int                uploaded_tiles     = 0;
#endif

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 7: loop principal

    
    using Clock = chrono::high_resolution_clock;
    auto then = Clock::now();

    int frame_count = 0;
    
    while(!glfwWindowShouldClose(window)){

        glfwPollEvents();
        
        // relança kernel se deu will_rerender

#if BH_CPU_BACKEND
        // CPU path: render em thread de fundo, tiles enviados progressivamente ao GL

        // faz upload dos tiles concluídos desde o último frame
        if (cpu_upload_pending) {
            TileProgress& tp = getTileProgress();

            // IMPORTANTE: ler render_done ANTES de ready.
            // Se render_done=true, a garantia seq_cst assegura que
            // tp.done já reflete todos os tiles — sem race condition.
            bool render_done = cpu_done.load();
            int  ready       = tp.done.load(std::memory_order_acquire);

            glBindTexture(GL_TEXTURE_2D, textures);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, config.WIDTH);
            while (uploaded_tiles < ready) {
                // tile_done_order[uploaded_tiles] diz qual tile físico terminou
                // nesta posição — corrige o problema de work-stealing fora de ordem.
                int tile_idx = tp.tile_done_order[uploaded_tiles];
                int tx = tile_idx % tp.tiles_x;
                int ty = tile_idx / tp.tiles_x;
                int x0 = tx * tp.tile_w;
                int y0 = ty * tp.tile_h;
                int x1 = std::min(x0 + tp.tile_w, config.WIDTH);
                int y1 = std::min(y0 + tp.tile_h, config.HEIGHT);
                glTexSubImage2D(GL_TEXTURE_2D, 0,
                                x0, y0, x1 - x0, y1 - y0,
                                GL_RGB, GL_UNSIGNED_BYTE,
                                pixels.data() + (y0 * config.WIDTH + x0) * 3);
                ++uploaded_tiles;
            }
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glBindTexture(GL_TEXTURE_2D, 0);

            if (render_done) {
                if (cpu_thread.joinable()) cpu_thread.join();
                // Safety-net: upload completo do frame para garantir que nenhum tile ficou faltando
                glBindTexture(GL_TEXTURE_2D, textures);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                config.WIDTH, config.HEIGHT,
                                GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
                glBindTexture(GL_TEXTURE_2D, 0);
                std::cout << "[render] completo — " << tp.total << " tiles\n";
                cpu_upload_pending = false;
                uploaded_tiles     = 0;
            }
        }

        // câmera mudou e não há render em andamento → lança thread
        if (camera.will_rerender && cpu_done.load()) {
            vec3 pos, fwd, right, up;
            camera.getVectors(pos, fwd, right, up);
            camera.will_rerender = false;
            cpu_done.store(false);
            cpu_upload_pending = true;
            uploaded_tiles     = 0;
            double3 c_pos   = { (double)pos.x,   (double)pos.y,   (double)pos.z   };
            double3 c_fwd   = { (double)fwd.x,   (double)fwd.y,   (double)fwd.z   };
            double3 c_right = { (double)right.x, (double)right.y, (double)right.z };
            double3 c_up    = { (double)up.x,    (double)up.y,    (double)up.z    };
            int   w   = config.WIDTH;
            int   h   = config.HEIGHT;
            float fov = config.fov_y;
            if (cpu_thread.joinable()) cpu_thread.join();
            cpu_thread = std::thread([=, &pixels, &cpu_done]() {
                launchRaytraceCPU(pixels.data(), w, h,
                                  c_pos, c_fwd, c_right, c_up,
                                  fov, rs_engine, starmap, perlin);
                cpu_done.store(true);
            });
        }

#else
        // CUDA backend: kernel escreve direto na textura GL via interop
        if (camera.will_rerender) {
            vec3 pos, fwd, right, up;
            camera.getVectors(pos, fwd, right, up);

            cudaGraphicsMapResources(1, &cuda_tex_resource, 0);

            cudaArray_t cuda_array;
            cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_tex_resource, 0, 0);

            raytraceCUDA(BH::is_gl, cuda_array,
                         config.WIDTH, config.HEIGHT,
                         pos, fwd, right, up,
                         config.fov_y);

            cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0);
            camera.will_rerender = false;
        }
#endif
        
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // 4 vértices, 2 triângulos → quad
        
        glfwSwapBuffers(window);

        frame_count++;
        auto now = Clock::now();

        float elapsed_time = chrono::duration<float>(now - then).count();
        if(elapsed_time >= 1.0f){

            float fps = frame_count / elapsed_time;
            string title = config.title + "  |  " + to_string(int(fps)) + " fps";
#if BH_CPU_BACKEND
            if (cpu_upload_pending) {
                TileProgress& tp = getTileProgress();
                int done_tiles = tp.done.load(std::memory_order_relaxed);
                int total_tiles = tp.total;
                title = config.title + "  |  renderizando " +
                        to_string(done_tiles) + "/" + to_string(total_tiles) +
                        " tiles  |  " + to_string(int(fps)) + " fps";
            }
#endif
            glfwSetWindowTitle(window, title.c_str());

            std::cout << "fps:" << fps << "\n";

            frame_count = 0;
            then = now;

        }
    }


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 8: limpeza dos dados se !glfwWindowShouldClose
    

#if BH_CPU_BACKEND
    getTileProgress().stop.store(true, std::memory_order_relaxed);
    if (cpu_thread.joinable()) cpu_thread.join();
#endif

    glDeleteTextures(1, &textures);
    glDeleteProgram(program);
    glDeleteVertexArrays(1, &VAO);

    glfwDestroyWindow(window);
    glfwTerminate(); 
    

}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
