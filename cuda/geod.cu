#include "../headers/geodesic.cuh"


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


__device__ void geodesicRHS(const Rays& s, double rhs[6], double rs) {

    // f = 1 - r_s/r  — fator de Schwarzschild.
    // Quando r >> r_s, f ≈ 1 (espaço plano). Em r = r_s, f = 0 (horizonte).
    
    double f = 1.0 - rs / s.r;
    if(s.r < rs * 1.1){

        double f_d = 1.0 - (double)rs / (double)s.r;
        f = (float)f_d;

    }

    //if (fabs(f) < 1e-6) f = (f < 0.00) ? -1e-6 : 1e-6;

    double dt    = s.E / f;

    double sin_t = sin(s.theta);
    double cos_t = cos(s.theta);
    double sin_t_safe = (fabs(sin_t) < 1e-12) ? 1e-12 : sin_t;

    // RHS[0,1,2] — triviais, vêm direto de Rays
    rhs[0] = s.dr;
    rhs[1] = s.dtheta;
    rhs[2] = s.dphi;


    // RHS[3] — aceleração radial
    rhs[3] =  - (rs / (2.0 * s.r * s.r)) * f * dt * dt
              + (rs / (2.0 * s.r * s.r * f)) * s.dr * s.dr
              + s.r * (s.dtheta * s.dtheta + sin_t * sin_t * s.dphi * s.dphi);


    // RHS[4] — aceleração angular polar (θ)
    rhs[4] =  - (2.0 / s.r) * s.dr * s.dtheta
              + sin_t * cos_t * s.dphi * s.dphi;


    // RHS[5] — aceleração angular azimutal (φ)
    rhs[5] =  - (2.0 / s.r) * s.dr * s.dphi
              - 2.0 * (cos_t / sin_t_safe) * s.dtheta * s.dphi;

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


__device__  void rk4Step(Rays& s, double dl, double rs){

    double y0[6] = { s.r, s.theta, s.phi, s.dr, s.dtheta, s.dphi };
    double k1[6], k2[6], k3[6], k4[6], tmp[6];
    Rays t;

    

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // k1: variação no ponto inicial
    geodesicRHS(s, k1, rs);

    

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // k2: variação num ponto médio estimado por k1

    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k1[i] * (dl / 2.0);

    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4]; t.dphi = tmp[5];

    geodesicRHS(t, k2, rs);
    
    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // k3: variação num ponto médio estimado por k2
    

    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k2[i] * (dl / 2.0);

    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4]; t.dphi = tmp[5];

    geodesicRHS(t, k3, rs);
    

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // k4: variação no ponto final estimado por k3


    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k3[i] * dl;

    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4]; t.dphi = tmp[5];

    geodesicRHS(t, k4, rs);
    

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // média final: (k1 + 2k2 + 2k3 + k4) / 6
    

    s.r      += (dl / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
    s.theta  += (dl / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
    s.phi    += (dl / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
    s.dr     += (dl / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);
    s.dtheta += (dl / 6.0) * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4]);
    s.dphi   += (dl / 6.0) * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5]);
    

}
//@}


// ─────────────────────────────────────────────────────────────────────────────────────────────────
