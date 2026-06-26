#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// disk_physics.hpp
//
// Porta das funções de effects.slang para C++ puro.
// Sem Vulkan, sem Slang, sem GLM — só física.
//
// Todos os valores de distância estão em unidades de RS (normalizado = 1.0),
// exatamente como no shader (constants.slang: RS = 1.0).
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <array>

namespace DiskPhysics {

// ─────────────────────────────────────────────────────────────────────────────
// Constantes — espelha constants.slang
// ─────────────────────────────────────────────────────────────────────────────

constexpr double RS              = 1.0;
constexpr double DISK_HEIGHT_SCALE = 0.025;
constexpr double DISK_R1_FACTOR  = 3.0;   // disco começa em 3 * RS (ISCO Schwarzschild)
constexpr double DISK_R2_FACTOR  = 13.0;  // disco termina em 13 * RS (você pediu até 13)

// ─────────────────────────────────────────────────────────────────────────────
// dopplerShift — porta fiel de effects.slang
//
//   v/c  = sqrt(rs / (2r - rs))
//   D    = 1 / (γ * (1 - β·cos_α))
//
// camera_pos: posição da câmera em unidades de RS
// phi:        ângulo azimutal do ponto no disco (rad)
// r_current:  raio equatorial do ponto (unidades de RS)
// ─────────────────────────────────────────────────────────────────────────────
inline double dopplerShift(double phi, double r_current,
                           double cam_x, double cam_y, double cam_z,
                           double rs = RS)
{
    double denom = 2.0 * r_current - rs;
    if (denom <= 0.0) return 1.0;

    double beta  = std::sqrt(rs / denom);
    double gamma = 1.0 / std::sqrt(1.0 - beta * beta);

    // velocidade orbital (tangencial, disco no equador)
    double vx = -std::sin(phi);
    double vy =  std::cos(phi);
    // vz = 0

    // vetor câmera → ponto
    double dx = cam_x - r_current * std::cos(phi);
    double dy = cam_y - r_current * std::sin(phi);
    double dz = cam_z;

    double dlen = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (dlen < 1e-10) return 1.0;

    dx /= dlen;
    dy /= dlen;

    double cos_alpha = vx * dx + vy * dy;

    return 1.0 / (gamma * (1.0 - beta * cos_alpha));
}

// ─────────────────────────────────────────────────────────────────────────────
// redShift — porta fiel de effects.slang
//
//   freq_shift = sqrt(1 - rs/r)
//   (inverso do fator gravitacional z = 1/sqrt(1 - rs/r))
// ─────────────────────────────────────────────────────────────────────────────
inline double redShift(double r_current, double rs = RS)
{
    if (r_current <= rs) return 0.0;
    return std::sqrt(1.0 - rs / r_current);
}

// ─────────────────────────────────────────────────────────────────────────────
// diskEmissivity — porta fiel de effects.slang
//
//   gaussiana vertical * perfil de temperatura radial
//   z = 0 (plano equatorial) → emissividade máxima
// ─────────────────────────────────────────────────────────────────────────────
inline double diskEmissivity(double r_current, double z,
                             double disk_r1, double disk_r2,
                             double height_scale = DISK_HEIGHT_SCALE)
{
    if (r_current < disk_r1 || r_current > disk_r2) return 0.0;

    double H        = r_current * height_scale;
    double gaussian = std::exp(-(z * z) / (2.0 * H * H));

    double t_norm       = (r_current - disk_r1) / (disk_r2 - disk_r1);
    double base         = 1.0 - t_norm * 0.8;
    double temp_profile = std::sqrt(base) * std::sqrt(std::sqrt(base)); // base^(3/4)

    return gaussian * temp_profile;
}

// ─────────────────────────────────────────────────────────────────────────────
// Temperatura física por anel — Novikov-Thorne simplificado
//
//   T(R) ∝ (R/R_isco)^(-3/4) * (1 - sqrt(R_isco/R))^(1/4)
//
// Retorna T normalizada [0, 1].
// ─────────────────────────────────────────────────────────────────────────────
inline double novikovThorneTemp(double r, double r_isco)
{
    if (r <= r_isco) return 0.0;
    double ratio   = r / r_isco;
    double profile = std::pow(ratio, -0.75) * std::pow(1.0 - std::sqrt(1.0 / ratio), 0.25);
    return profile; // normalizado: pico próximo de r_isco
}

// ─────────────────────────────────────────────────────────────────────────────
// Luminosidade por anel — L(R) dR
//
//   L ∝ emissividade * 2πR * dR   (anel de largura dR)
//
// Integra z no plano equatorial (z = 0).
// ─────────────────────────────────────────────────────────────────────────────
inline double ringLuminosity(double r, double dr,
                             double disk_r1, double disk_r2)
{
    constexpr double PI = 3.14159265358979323846;
    double emiss = diskEmissivity(r, 0.0, disk_r1, disk_r2);
    return emiss * 2.0 * PI * r * dr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Struct que representa os dados de um anel
// ─────────────────────────────────────────────────────────────────────────────
struct RingData {
    double r;               // raio do anel em unidades de RS
    double r_phys;          // raio físico em metros (r * RS_phys)
    double emissivity;      // emissividade no plano equatorial
    double temp_norm;       // temperatura normalizada (0 = fria, 1 = quente)
    double redshift;        // fator de redshift gravitacional
    double doppler_front;   // doppler no lado frontal (se aproximando)
    double doppler_back;    // doppler no lado traseiro (se afastando)
    double luminosity;      // luminosidade do anel (emissividade * área)
    double energy_obs;      // energia observada (emissividade * doppler_frente * redshift)
};

} // namespace DiskPhysics