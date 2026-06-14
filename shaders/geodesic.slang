#pragma once

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// struct Rays:
// estado completo de um raio em coordenadas esféricas de Schwarzschild

struct Rays {

    //  coordenadas esféricas:
    //  distância radial ao centro do buraco negro:  r     (metros)
    //  ângulo polar relativo ao polo norte:         theta (radianos, [0, π])
    //  ângulo azimutal relativo ao equador:         phi   (radianos, [0, 2π])
    float r, theta, phi;

    //  primeira derivada em relação ao parâmetro afim λ:
    //  ∂r/∂λ, ∂θ/∂λ, ∂φ/∂λ
    float dr, dtheta, dphi;

    //  constantes de movimento (conservadas ao longo da geodésica):
    //  E → energia por unidade de massa
    //  L → momento angular por unidade de massa
    float E, L;
    
    bool is_trapped;

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//geodesicRHS:
//@{
/* 
    geodesicRHS — calcula as derivadas da geodésica de Schwarzschild

    equação:  d²xᵘ/dλ² + Γᵘ_αβ (dxᵅ/dλ)(dx^β/dλ) = 0
        → dados os estados atuais, computa as derivadas relativas a esse estado.

    Γ é o símbolo de Christoffel da métrica de Schwarzschild.
    Na expansão em 3 dimensões, resultam 6 EDOs:
        d/dλ [ r, θ, φ, dr, dθ, dφ ] = rhs[0..5]
*/ 

void geodesicRHS(inout Rays s, out float rhs[6], float rs) {

    // Hard-stop calculations if the ray is already trapped
    if (s.is_trapped || s.r <= rs || !isfinite(s.r)) {
        s.is_trapped = true;
        for (int i = 0; i < 6; i++) {
            rhs[i] = 0.0f;
        }
        return;
    }

    // f = 1 - r_s/r  — fator de Schwarzschild.
    // Quando r >> r_s, f ≈ 1 (espaço plano). Em r = r_s, f = 0 (horizonte).
    float f = 1.0f - rs / s.r;
    
    // Clamp f near the horizon to prevent float-precision dt blowup.
    // In float, f < 1e-6 makes dt = E/f explode, causing rhs[3] to go
    // unstable and the ray to be falsely captured.
    if (s.r < rs * 1.0001f) {
        f = 1.0f - rs / (rs * 1.0001f);  // clamps f to ~1e-4
    }

    float dt    = s.E / f;
    float sin_t = sin(s.theta);
    float cos_t = cos(s.theta);
    float sin_t_safe = (fabs(sin_t) < 1e-4f) ? (sin_t < 0.0f ? -1e-4f : 1e-4f) : sin_t;
    float sin_phi_term = (fabs(sin_t) < 1e-4f) ? 0.0f : (sin_t * sin_t * s.dphi * s.dphi);

    // RHS[0,1,2] — triviais, vêm direto de Rays
    rhs[0] = s.dr;
    rhs[1] = s.dtheta;
    rhs[2] = s.dphi;

    // RHS[3] — aceleração radial
    rhs[3] =    - (rs / (2.0f * s.r * s.r)) * f * dt * dt
                + (rs / (2.0f * s.r * s.r * f)) * s.dr * s.dr
                + s.r * (s.dtheta * s.dtheta + sin_phi_term);

    // RHS[4] — aceleração angular polar (θ)
    rhs[4] =    - (2.0f / s.r) * s.dr * s.dtheta
                + sin_t * cos_t * s.dphi * s.dphi;

    // RHS[5] — aceleração angular azimutal (φ)
    if (fabs(sin_t) < 1e-4f) {
        rhs[5] = 0.0f;
    } else {
        rhs[5] = - (2.0f / s.r) * s.dr * s.dphi
                 - 2.0f * (cos_t / sin_t_safe) * s.dtheta * s.dphi;
    }
        
    //rhs[5] = 0.0;   

}
//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// rk4Step:
//@{
/* 
    rk4Step — integrador Runge-Kutta de 4ª ordem:

    Resolve as 6 EDOs acima em um passo de tamanho dl.
    Faz uma média ponderada de 4 avaliações da derivada em pontos distintos,
    o que reduz o erro de truncamento para O(dl⁵) por passo.

    k1 → derivada no ponto inicial
    k2 → derivada no ponto médio estimado por k1
    k3 → derivada no ponto médio estimado por k2
    k4 → derivada no ponto final estimado por k3
    média final: state += (dl/6) * (k1 + 2k2 + 2k3 + k4)

*/

float reconstructDphi(float L, float r, float sin_t) {
    float sin_t_safe = (fabs(sin_t) < 1e-6f) ? 1e-6f : sin_t;
    return L / (r * r * sin_t_safe * sin_t_safe);
}

void rk4Step(inout Rays s, float dl, float rs) {
    if (s.is_trapped) return;

    // Reconstruct dphi from conserved L to erase any float drift accumulated so far
    s.dphi = reconstructDphi(s.L, s.r, sin(s.theta));

    float y0[6] = { s.r, s.theta, s.phi, s.dr, s.dtheta, s.dphi };
    float k1[6], k2[6], k3[6], k4[6], tmp[6];
    Rays t;

    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // k1: variação no ponto inicial
    geodesicRHS(s, k1, rs);
    if (s.is_trapped) return;

    

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // k2: variação num ponto médio estimado por k1
    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k1[i] * (dl / 2.0f);
    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4];
    t.dphi = reconstructDphi(t.L, t.r, sin(t.theta));
    geodesicRHS(t, k2, rs);
    if (t.is_trapped) {
        s.is_trapped = true;
        return;
    }
    
    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // k3: variação num ponto médio estimado por k2
    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k2[i] * (dl / 2.0f);
    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4];
    t.dphi = reconstructDphi(t.L, t.r, sin(t.theta));
    geodesicRHS(t, k3, rs);
    if (t.is_trapped) {
        s.is_trapped = true;
        return;
    }
    

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // k4: variação no ponto final estimado por k3
    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k3[i] * dl;
    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4];
    t.dphi = reconstructDphi(t.L, t.r, sin(t.theta));
    geodesicRHS(t, k4, rs);
    if (t.is_trapped) {
        s.is_trapped = true;
        return;
    }
    

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // média final: (k1 + 2k2 + 2k3 + k4) / 6
    float next_r = s.r + (dl / 6.0f) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
    
    // Cleanly trap rays that cross the critical threshold, preventing NaN propagating onwards
    if (next_r <= rs || !isfinite(next_r)) {
        s.is_trapped = true;
        return;
    }

    s.r      = next_r;
    s.theta  += (dl / 6.0f) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
    s.phi    += (dl / 6.0f) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
    s.dr     += (dl / 6.0f) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);
    s.dtheta += (dl / 6.0f) * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4]);
    // Reconstruct dphi from conserved L to prevent float drift
    s.dphi   = reconstructDphi(s.L, s.r, sin(s.theta));
    

}
//@}


// ─────────────────────────────────────────────────────────────────────────────────────────────────
