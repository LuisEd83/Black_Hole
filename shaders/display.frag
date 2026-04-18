#version 330 core

in  vec2 uv;
out vec4 fragColor;

// textura RGB escrita pelo kernel CUDA a cada frame
uniform sampler2D u_frame;

void main() {
    fragColor = texture(u_frame, uv);
}
