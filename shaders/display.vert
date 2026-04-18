#version 330 core

// quad fullscreen em NDC — nenhum VBO necessário.
// os 4 vértices são gerados pelo gl_VertexID, chamado com glDrawArrays(..., 0, 4).

out vec2 uv;

void main() {
    vec2 pos = vec2(
        (gl_VertexID & 1) == 0 ? -1.0 : 1.0,
        (gl_VertexID & 2) == 0 ? -1.0 :  1.0
    );
    uv = pos * 0.5 + 0.5;      // [-1,1] → [0,1]
    gl_Position = vec4(pos, 0.0, 1.0);
}
