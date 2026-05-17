/*
    cpu_raytrace.cpp — CPU backend for macOS (BH_CPU_BACKEND=1).

    Full port of the CUDA ray tracer from cuda/geodesic.cu.
    Parallelism: one std::thread per logical core, each handles a horizontal
    band of pixels.

    Algorithm is identical to the GPU kernel: Schwarzschild geodesic
    integration via RK4, volumetric accretion disk, Doppler shift, redshift.
*/

#include "cpu_raytrace.hpp"
#include "cpu_texture.hpp"
#include "constants.hpp"

#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ── Physical constants (mirrors geodesic.cuh) ────────────────────────────────

static const double kC    = 299792458.0;
static const double kG    = 6.67430e-11;
static const double kMass = 8.54e36;
static const double kRS   = 2.0 * kG * kMass / (kC * kC);



// ── Ray state (mirrors struct Rays in geodesic.cuh) ──────────────────────────

struct Ray {
    double r, theta, phi;
    double dr, dtheta, dphi;
    double E, L;
};


// ── Geodesic RHS ─────────────────────────────────────────────────────────────

static void geodesicRHS(const Ray& s, double rhs[6], double rs) {
    double f  = 1.0 - rs / s.r;
    if (std::fabs(f) < 1e-6) f = (f < 0.0) ? -1e-6 : 1e-6;

    double dt    = s.E / f;
    double sin_t = std::sin(s.theta);
    double cos_t = std::cos(s.theta);
    double st    = (std::fabs(sin_t) < 1e-12) ? 1e-12 : sin_t;

    rhs[0] = s.dr;
    rhs[1] = s.dtheta;
    rhs[2] = s.dphi;
    rhs[3] = -(rs / (2.0*s.r*s.r)) * f * dt*dt
             + (rs / (2.0*s.r*s.r*f)) * s.dr*s.dr
             + s.r * (s.dtheta*s.dtheta + sin_t*sin_t * s.dphi*s.dphi);
    rhs[4] = -(2.0/s.r) * s.dr*s.dtheta + sin_t*cos_t * s.dphi*s.dphi;
    rhs[5] = -(2.0/s.r) * s.dr*s.dphi   - 2.0*(cos_t/st) * s.dtheta*s.dphi;
}


// ── RK4 step ─────────────────────────────────────────────────────────────────

static void rk4Step(Ray& s, double dl, double rs) {
    double y0[6] = { s.r, s.theta, s.phi, s.dr, s.dtheta, s.dphi };
    double k1[6], k2[6], k3[6], k4[6], tmp[6];
    Ray t;

    geodesicRHS(s, k1, rs);

    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k1[i]*(dl/2.0);
    t.r=tmp[0]; t.theta=tmp[1]; t.phi=tmp[2];
    t.dr=tmp[3]; t.dtheta=tmp[4]; t.dphi=tmp[5];
    geodesicRHS(t, k2, rs);

    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k2[i]*(dl/2.0);
    t.r=tmp[0]; t.theta=tmp[1]; t.phi=tmp[2];
    t.dr=tmp[3]; t.dtheta=tmp[4]; t.dphi=tmp[5];
    geodesicRHS(t, k3, rs);

    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k3[i]*dl;
    t.r=tmp[0]; t.theta=tmp[1]; t.phi=tmp[2];
    t.dr=tmp[3]; t.dtheta=tmp[4]; t.dphi=tmp[5];
    geodesicRHS(t, k4, rs);

    s.r      += (dl/6.0)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);
    s.theta  += (dl/6.0)*(k1[1]+2*k2[1]+2*k3[1]+k4[1]);
    s.phi    += (dl/6.0)*(k1[2]+2*k2[2]+2*k3[2]+k4[2]);
    s.dr     += (dl/6.0)*(k1[3]+2*k2[3]+2*k3[3]+k4[3]);
    s.dtheta += (dl/6.0)*(k1[4]+2*k2[4]+2*k3[4]+k4[4]);
    s.dphi   += (dl/6.0)*(k1[5]+2*k2[5]+2*k3[5]+k4[5]);
}


// ── Physical effects (mirrors geodesic.cu) ───────────────────────────────────

static float dopplerShift(double r_current, double phi,
                           double3 camera_pos, double rs) {
    double denom = 2.0*r_current - rs;
    if (denom <= 0.0) return 1.0f;

    double beta  = std::sqrt(rs / denom);
    double gamma = 1.0 / std::sqrt(1.0 - beta*beta);

    double vx = -std::sin(phi);
    double vy =  std::cos(phi);

    double dx = camera_pos.x - r_current*std::cos(phi);
    double dy = camera_pos.y - r_current*std::sin(phi);
    double dz = camera_pos.z;
    double dl = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (dl < 1e-10) return 1.0f;
    dx /= dl; dy /= dl;

    double cos_alpha = vx*dx + vy*dy;
    return (float)(1.0 / (gamma * (1.0 - beta*cos_alpha)));
}

static float redShift(double r_current, double rs) {
    double z = 1.0 / std::sqrt(1.0 - rs/r_current);
    return (float)(1.0 / z);
}

static float diskEmissivity(double r_current, double z_cart,
                             double disk_r1, double disk_r2,
                             double height_scale) {
    if (r_current < disk_r1 || r_current > disk_r2) return 0.f;
    double H        = r_current * height_scale;
    double gaussian = std::exp(-(z_cart*z_cart) / (2.0*H*H));
    double t_norm   = (r_current - disk_r1) / (disk_r2 - disk_r1);
    double temp     = std::pow(1.0 - t_norm*0.8, 0.75);
    return (float)(gaussian * temp);
}

static float perlinSample(CpuTexture* tex,
                           double r_current, double phi,
                           double disk_r1, double disk_r2) {
    if (!tex) return 1.0f;
    float u = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));
    float v = (float)(phi / (2.0*M_PI)) + 0.5f;
    return tex->sample(u, v).x;
}

static void temperatureToColor(float t, float& r, float& g, float& b) {
    if (t > 0.8f) {
        float s = (t - 0.8f) / 0.2f;
        r = 1.f; g = 1.f; b = 0.8f + 0.2f*s;
    } else if (t > 0.5f) {
        float s = (t - 0.5f) / 0.3f;
        r = 1.f; g = 0.7f + 0.3f*s; b = 0.1f + 0.7f*s;
    } else if (t > 0.2f) {
        float s = (t - 0.2f) / 0.3f;
        r = 1.f; g = 0.3f + 0.4f*s; b = 0.0f + 0.1f*s;
    } else {
        float s = t / 0.2f;
        r = 0.6f + 0.4f*s; g = 0.05f + 0.25f*s; b = 0.f;
    }
}


// ── Single-pixel ray tracer ───────────────────────────────────────────────────

static void processPixel(int x, int y, int WIDTH, int HEIGHT,
                          unsigned char* pixels,
                          double3 cam_pos, double3 cam_fwd,
                          double3 cam_right, double3 cam_up,
                          float fov_y, double rs,
                          CpuTexture* starmap_tex, CpuTexture* perlin_tex,
                          const std::atomic<bool>& stop) {

    float aspect     = (float)WIDTH / (float)HEIGHT;
    float tanHalfFov = std::tan(fov_y * 0.5f * (float)M_PI / 180.f);
    float u_sc       = (2.f*(x+0.5f)/WIDTH  - 1.f) * aspect * tanHalfFov;
    float v_sc       = (1.f - 2.f*(y+0.5f)/HEIGHT) * tanHalfFov;

    double dx = u_sc*cam_right.x + v_sc*cam_up.x + cam_fwd.x;
    double dy = u_sc*cam_right.y + v_sc*cam_up.y + cam_fwd.y;
    double dz = u_sc*cam_right.z + v_sc*cam_up.z + cam_fwd.z;
    double dlen = std::sqrt(dx*dx + dy*dy + dz*dz);
    dx /= dlen; dy /= dlen; dz /= dlen;

    double ox = cam_pos.x, oy = cam_pos.y, oz = cam_pos.z;
    double r0     = std::sqrt(ox*ox + oy*oy + oz*oz);
    double theta0 = std::acos(std::fmax(-1.0, std::fmin(1.0, oz/r0)));
    double phi0   = std::atan2(oy, ox);
    if (theta0 > M_PI - 1e-6) theta0 = M_PI - 1e-6;

    double st = std::sin(theta0), ct = std::cos(theta0);
    double sp = std::sin(phi0),   cp = std::cos(phi0);
    double st_safe = (std::fabs(st) < 1e-12) ? 1e-12 : st;

    Ray s;
    s.r      = r0;
    s.theta  = theta0;
    s.phi    = phi0;
    s.dr     =  st*cp*dx + st*sp*dy + ct*dz;
    s.dtheta = (ct*cp*dx + ct*sp*dy - st*dz) / r0;
    s.dphi   = (-sp*dx + cp*dy) / (r0*st_safe);

    double f0   = 1.0 - rs/r0;
    double vmag = std::sqrt(s.dr*s.dr/f0
                            + r0*r0*s.dtheta*s.dtheta
                            + r0*r0*st_safe*st_safe*s.dphi*s.dphi);
    s.L = r0*r0*st_safe*s.dphi;
    s.E = f0*vmag;

    // ── simulation parameters (mirrors geodesic.cu) ──────────────────────────
    const double step          = rs * BH::STEP_FACTOR;
    const double escape_radius = rs * 20.0;
    const double disk_r1       = rs * 3.0;
    const double disk_r2       = rs * 12.0;
    const double adaptive_clamp  = 0.001;
    const double disk_height_scale = 0.08;
    const double disk_opacity    = 4.0;
    const double emission_scale  = 2.5;
    const double closeness       = 5.0;

    float accum_r = 0.f, accum_g = 0.f, accum_b = 0.f, accum_alpha = 0.f;
    unsigned char R = 0, G = 0, B = 0;

    enum class Result { NONE, HORIZON, ESCAPE, DISK, FALLBACK };
    Result result = Result::NONE;

    auto sampleStar = [&]() {
        while (s.phi >  M_PI) s.phi -= 2.0*M_PI;
        while (s.phi < -M_PI) s.phi += 2.0*M_PI;
        float ut = (float)(s.phi/(2.0*M_PI)) + 0.5f;
        float vt = (float)(s.theta/M_PI);
        float4 col = starmap_tex ? starmap_tex->sample(ut, vt) : float4{0,0,0,1};
        R = (unsigned char)std::fmin(col.x*255.f, 255.f);
        G = (unsigned char)std::fmin(col.y*255.f, 255.f);
        B = (unsigned char)std::fmin(col.z*255.f, 255.f);
    };

    // ── impact-parameter early exit ───────────────────────────────────────────
    // b_crit = 3√3/2 · rs: raios com b < b_crit são capturados pelo BH.
    const double b_crit = 2.598 * rs;
    const double b_init = (s.E > 1e-10) ? std::fabs(s.L / s.E) : 1e30;
    const bool definitely_captured = (b_init < b_crit);

    {
        if (b_init > b_crit*BH::IMPACT_CUTOFF && s.r > disk_r2*1.5 && s.dr > 0.0) {
            sampleStar();
            result = Result::ESCAPE;
            goto write_pixel;
        }
    }

    // ── integration loop ──────────────────────────────────────────────────────
    for (int i = 0; i < BH::MAX_STEPS; ++i) {

        if (stop.load(std::memory_order_relaxed)) goto write_pixel;

        if (s.r <= rs || s.r <= 0.0) {
            R = G = B = 0;
            result = Result::HORIZON;
            break;
        }

        if (s.r > escape_radius) {
            sampleStar();
            result = Result::ESCAPE;
            break;
        }

        // adaptive step
        double adaptive_step = step;
        if (s.r < BH::ADAPTIVE_FACTOR * rs) {
            adaptive_step = step * (s.r / (rs * BH::ADAPTIVE_FACTOR));
            if (adaptive_step < step * adaptive_clamp)
                adaptive_step = step * adaptive_clamp;
        }

        rk4Step(s, adaptive_step, rs);

        if (s.r <= rs || s.r <= 0.0 || s.r != s.r) {
            R = G = B = 0;
            result = Result::HORIZON;
            break;
        }

        // early exit: raio escapou do disco, usa break (não goto) para preservar
        // o blending de emissão de disco que pode ter sido acumulado
        if (s.r > disk_r2*1.5 && s.dr > 0.0) {
            result = Result::ESCAPE;
            break;
        }

        // Raios capturados (b < b_crit) abaixo de disk_r1 não acumulam mais disco
        // e inevitavelmente caem no horizonte — sai com preto sem percorrer os 5000 passos.
        if (definitely_captured && s.r < disk_r1) {
            R = G = B = 0;
            result = Result::HORIZON;
            break;
        }

        // disk accumulation
        double xc = s.r*std::sin(s.theta)*std::cos(s.phi);
        double yc = s.r*std::sin(s.theta)*std::sin(s.phi);
        double zc = s.r*std::cos(s.theta);
        double r_eq = std::sqrt(xc*xc + yc*yc);

        float emiss = diskEmissivity(r_eq, zc, disk_r1, disk_r2, disk_height_scale);
        if (emiss > BH::EMISSIVITY_RATE) {
            float doppler  = dopplerShift(r_eq, s.phi, cam_pos, rs);
            float rs_freq  = redShift(r_eq, rs);
            float noise    = perlinSample(perlin_tex, r_eq, s.phi, disk_r1, disk_r2);
            float base_b   = 0.4f + 0.6f*noise;

            float freq_effect = doppler * rs_freq;
            float beam_effect = std::pow(std::fmax(doppler, 0.01f), 3.f);

            float temp_norm = (float)std::pow(1.0 - ((r_eq-disk_r1)/(disk_r2-disk_r1))*0.8, 0.75);
            float dr, dg, db;
            temperatureToColor(temp_norm, dr, dg, db);

            if (freq_effect > 1.f) {
                float blend = std::fmin((freq_effect-1.f)/1.5f, 1.f);
                dg += blend*(1.f-dg);
                db += blend*(1.f-db);
            } else {
                float blend = std::fmin((1.f-freq_effect)/0.8f, 1.f);
                dr += blend*0.3f;
                dg += blend*(1.f-dg);
                db += blend*(1.f-db);
            }

            float intensity     = emiss * base_b * (float)emission_scale * beam_effect;
            float step_norm     = (float)(adaptive_step / (disk_r2 - disk_r1));
            float contribution  = intensity * step_norm * (float)disk_opacity;
            float remaining     = 1.f - accum_alpha;

            accum_r     += dr * contribution * remaining;
            accum_g     += dg * contribution * remaining;
            accum_b     += db * contribution * remaining;
            accum_alpha += contribution * remaining;

            result = Result::DISK;

            if (accum_alpha >= 1.f) { accum_alpha = 1.f; break; }
        }
    }

    // ── fallback ──────────────────────────────────────────────────────────────
    if (result == Result::NONE) result = Result::FALLBACK;

    if (result == Result::HORIZON) {
        R = G = B = 0;
    } else if (result == Result::ESCAPE || result == Result::FALLBACK) {
        if (result == Result::FALLBACK && s.r < rs * closeness) {
            R = G = B = 0;
        } else {
            sampleStar();
        }
    }

    if (accum_alpha > 0.0001f) {
        float bg = 1.f - std::fmin(accum_alpha, 1.f);
        R = (unsigned char)std::fmin((accum_r + R/255.f*bg)*255.f, 255.f);
        G = (unsigned char)std::fmin((accum_g + G/255.f*bg)*255.f, 255.f);
        B = (unsigned char)std::fmin((accum_b + B/255.f*bg)*255.f, 255.f);
    }

write_pixel:
    int idx = (y * WIDTH + x) * 3;
    pixels[idx+0] = R;
    pixels[idx+1] = G;
    pixels[idx+2] = B;
}


// ── Global tile progress (lido pelo engine para upload progressivo) ───────────

static TileProgress g_tile_progress;
TileProgress& getTileProgress() { return g_tile_progress; }


// ── Work-stealing tile launcher ───────────────────────────────────────────────
// Divide a imagem em tiles de TW×TH pixels. Cada thread compete por tiles via
// atomic, sem fila explícita. Ao concluir cada tile, incrementa g_tile_progress.done
// com memory_order_release para que o engine veja os pixels escritos antes do count.

void launchRaytraceCPU(void* pixels_raw, int W, int H,
                        double3 pos, double3 fwd, double3 right, double3 up,
                        float fov_y, double rs,
                        cudaTextureObject_t starmap_h, cudaTextureObject_t perlin_h) {

    auto* pixels   = static_cast<unsigned char*>(pixels_raw);
    auto* smap_tex = cpu_tex(starmap_h);
    auto* perl_tex = cpu_tex(perlin_h);

    constexpr int TW = 64, TH = 64;
    int tiles_x = (W + TW - 1) / TW;
    int tiles_y = (H + TH - 1) / TH;
    int total   = tiles_x * tiles_y;

    g_tile_progress.tiles_x = tiles_x;
    g_tile_progress.tile_w  = TW;
    g_tile_progress.tile_h  = TH;
    g_tile_progress.total   = total;
    g_tile_progress.done.store(0, std::memory_order_relaxed);
    g_tile_progress.next_pos.store(0, std::memory_order_relaxed);
    g_tile_progress.stop.store(false, std::memory_order_relaxed);
    g_tile_progress.tile_done_order.resize(total);

    std::atomic<int> next_tile{0};
    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers(nthreads);

    for (auto& worker : workers) {
        worker = std::thread([&]() {
            while (true) {
                if (g_tile_progress.stop.load(std::memory_order_relaxed)) break;
                int tile = next_tile.fetch_add(1, std::memory_order_relaxed);
                if (tile >= total) break;
                int tx = tile % tiles_x, ty = tile / tiles_x;
                int x0 = tx * TW,  y0 = ty * TH;
                int x1 = std::min(x0 + TW, W);
                int y1 = std::min(y0 + TH, H);
                for (int y = y0; y < y1; ++y)
                    for (int x = x0; x < x1; ++x)
                        processPixel(x, y, W, H, pixels,
                                     pos, fwd, right, up,
                                     fov_y, rs, smap_tex, perl_tex,
                                     g_tile_progress.stop);
                // Registra qual tile terminou nesta posição, depois sinaliza.
                // A release sequence em `done` garante que o main thread vê
                // tile_done_order[0..done-1] quando lê done com acquire.
                int order_pos = g_tile_progress.next_pos.fetch_add(1, std::memory_order_relaxed);
                g_tile_progress.tile_done_order[order_pos] = tile;
                g_tile_progress.done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    for (auto& w : workers) w.join();
}


// ── Stubs for CUDA-only helpers ───────────────────────────────────────────────

void activateSetFlags() {}  // no-op: cudaDeviceScheduleBlockingSync not needed

unsigned int* getStateCountsPtr() { return nullptr; }
