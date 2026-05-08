#include "engine.hpp"
#include "constants.hpp"
 
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

 
#include <iostream>
#include <surface_types.h>
#include <texture_types.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

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

 

void launchGL(  cudaSurfaceObject_t surface,
                int WIDTH, 
                int HEIGHT,
                double3 pos,
                double3 fwd,
                double3 right,
                double3 up,
                float fov_y,
                double rs,
                cudaTextureObject_t starmap,
                cudaTextureObject_t perlin);
       

void raytraceCUDA(  unsigned char* pixels,
                    cudaSurfaceObject_t surface,
                    int WIDTH, 
                    int HEIGHT,
                    vec3 pos, 
                    vec3 fwd, 
                    vec3 right, 
                    vec3 up,
                    float fov_y);



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//@{


static cudaGraphicsResource_t cuda_tex_resource = nullptr;

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

//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//@{


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

//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//@{


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

//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//@{


static void mouseButtons(GLFWwindow* window, int button, int action, int){

    /*
        processando eventual click do mouse: filtra drag como o press do botão esquerdo.   
    */
        
    
    SimConfig* config = (SimConfig*)glfwGetWindowUserPointer(window);

    if(button == GLFW_MOUSE_BUTTON_LEFT){ config->dragging = (action == GLFW_PRESS); }
}

static void cursorPos(GLFWwindow* window, double x, double y){
    
    /*
    
    */
    
    SimConfig* config = (SimConfig*)glfwGetWindowUserPointer(window);

    const float sensitivity = 10e-10f;

    if(!config->dragging){
        
        config->last_mouse_x = x;
        config->last_mouse_y = y;

        return;
    }

    float dx = float(x - config->last_mouse_x);
    float dy = float(y - config->last_mouse_y);

    config->last_mouse_x = x;
    config->last_mouse_y = y;
    

    config->azimuth += dx * sensitivity;
    config->elevation += dy * sensitivity;
    
    // apenas uma garantia para a câmera não poder virar de cabeça pra baixo
    const float limit = radians(89.0f);
    if(config->elevation > limit) config->elevation = limit;
    if(config->elevation < -limit) config->elevation = -limit;
    
    config->use_direct_vectors = false;
    config->will_rerender = true;
}


static void scrollEvent(GLFWwindow* window, double, double y_off){

    SimConfig* config = (SimConfig*)glfwGetWindowUserPointer(window);

    config->cam_dist -= float(y_off) * config->cam_dist * 0.0f;
    if(config->cam_dist < 1.0f) config->cam_dist = 1.0f;

    config->will_rerender = true;
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

//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//@{ 


void engineRun(SimConfig& config){


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 1: iniciar GLFW
    //@{

    //glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        
    if(!glfwInit()){
        cerr << "[ENGINE]: Falha ao iniciar GLFW.\n";
        return;
    }
    
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(config.WIDTH, config.HEIGHT, config.title.c_str(), nullptr, nullptr);

    if(!window){
        cerr << "[ENGINE]: Falha ao criar janela GLFW.\n";         
        glfwTerminate();

        return;
    }

    glfwSetWindowUserPointer(window, &config);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); 
    
    //@}

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 2: iniciar GLEW
    //@{ 
    

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){
        cerr << "[ENGINE]: Falha ao iniciar GLEW.\n";
        
        glfwDestroyWindow(window);
        glfwTerminate();

        return;
    }

    //@}

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 3: coordenação de IO (callbacks)
    //@{ 


    glfwSetMouseButtonCallback(window, mouseButtons);
    glfwSetCursorPosCallback(window, cursorPos);
    glfwSetScrollCallback(window, scrollEvent);
    glfwSetKeyCallback(window, keypressEvent);
    glfwSetFramebufferSizeCallback(window, frameBuffer);

    std::cout << "Callbacks Inicializados.\n";
    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 4: settar estado inicial da camera
    //@{

    vec3 dir = normalize(config.pos - config.target);
    config.elevation = asin(dir.z);
    config.azimuth = atan2(dir.y, dir.x);
    config.will_rerender = true;
    
    std::cout << "Camera Inicial Aplicada.\n";
    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 5: recursos do OpenGL
    //@{

        
    string vert = loadShaders("shaders/display.vert");
    string frag = loadShaders("shaders/display.frag");


    const char* vert_str = vert.c_str();
    const char* frag_str = frag.c_str();
    
    GLuint program  = buildProgram(vert_str, frag_str);
    if(program == 0){
        cerr << "[ENGINE]: Falha ao compilar shaders.\n";
        return;
    }

    GLuint textures = makeGLFrameTexture(config.WIDTH,  config.HEIGHT);
    
    cudaGraphicsGLRegisterImage(&cuda_tex_resource, textures,
                             GL_TEXTURE_2D,
                             cudaGraphicsRegisterFlagsSurfaceLoadStore);
    
    ck("RegisterImage");

    // criaçao do VAO vazio
    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_frame"), 0); 
    // textura unit 0. funciona como um índice, você pode armazenar várias texturas.
    // nesse caso, temos apenas uma, logo índice = 0. 
        
    std::cout << "Recursos Aplicados.\n";

     //@}
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 6: reservar buffer de pixels para __host__
    //@{


    vector<unsigned char>pixels;
    pixels.resize(config.WIDTH * config.HEIGHT * 3); // 3 canais de RGB
    
    
    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 7: loop principal  
    //@{
    

    using Clock = chrono::high_resolution_clock;
    auto then = Clock::now();

    int frame_count = 0;
    
    while(!glfwWindowShouldClose(window)){

        glfwPollEvents();
                    
        // relança kernel se deu will_rerender
        if(config.will_rerender){

            vec3 pos, fwd, right, up;

           /*
            if(config.use_direct_vectors && frame_count == 0){

                pos   = config.cam_pos;
                fwd   = config.cam_fwd;
                right = config.cam_right;
                up    = config.cam_up_vec;

            } else {
            }
            */


            config.getVectors(pos, fwd, right, up);


            // mapeia textura GL para CUDA
            glFinish(); 

            cudaGraphicsMapResources(1, &cuda_tex_resource, 0);
            ck("MapResources");

            cudaArray_t cuda_array;
            cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_tex_resource, 0, 0);
            ck("GetMappedArray");

            cudaResourceDesc res_desc = {};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = cuda_array;

            cudaSurfaceObject_t surface;
            cudaCreateSurfaceObject(&surface, &res_desc);
            ck("CreateSurface");

            std::cout << "Kernel Iniciado.\n";
               
            
            double3 c_pos   = { pos.x,   pos.y,   pos.z   };
            double3 c_fwd   = { fwd.x,   fwd.y,   fwd.z   };
            double3 c_right = { right.x, right.y, right.z };
            double3 c_up    = { up.x,    up.y,    up.z    };
            
/*

            launchGL(   surface,
                        config.WIDTH,
                        config.HEIGHT,
                        c_pos,
                        c_fwd,
                        c_right,
                        c_up,
                        config.fov_y,
                        RS,
                        config.starmap,
                        config.perlin
                     );
           
*/

           raytraceCUDA(NULL,
                        surface,
                        config.WIDTH,
                        config.HEIGHT,
                        pos,
                        fwd,
                        right,
                        up,
                        config.fov_y
                     );



            cudaStreamSynchronize(0);

            ck("PostKernel");

            // desmapeia antes do GL desenhar
            cudaDestroySurfaceObject(surface);
            ck("DestroySurface");

            cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0);
            ck("UnmapResources"); 
            
            cout << "─────────────────────────────────────────────────────────────────────────────────────────────────\n"; 

            //glBindTexture(GL_TEXTURE_2D, textures);
            //glFinish();  // ← wait for GL to sync
            
            
            // sem glTexSubImage2D — kernel já escreveu na textura
            config.will_rerender = false;
        }


        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // 4 vértices, 2 triângulos → quad
        
        glfwSwapBuffers(window);

        frame_count++;
        auto now = Clock::now();

        //std::cout << frame_count << "\n";

        float elapsed_time = chrono::duration<float>(now - then).count();
        if(elapsed_time >= 1.0f){

            float fps = frame_count / elapsed_time;
            string title = config.title + "  |  " + to_string(int(fps)) + " fps";
            glfwSetWindowTitle(window, title.c_str());
        
            std::cout << "fps:" << fps << "\n";

            frame_count = 0;
            then = now;

        }
    }

    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // passo 8: limpeza dos dados se !glfwWindowShouldClose
    //@{
    

    glDeleteTextures(1, &textures);
    glDeleteProgram(program);
    glDeleteVertexArrays(1, &VAO);

    glfwDestroyWindow(window);
    glfwTerminate(); 
    
    //@}


}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
