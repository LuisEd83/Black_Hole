#include "constants.slang"
#include "geodesic.slang"
#include "effects.slang"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Descriptor bindings
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[[vk::binding(0, 0)]]
RWTexture2D<float4> outputImage;

[[vk::binding(1, 0)]]
ConstantBuffer<RenderParams> params;

[[vk::binding(2, 0)]]
Sampler2D starmap;

[[vk::binding(3, 0)]]
Sampler3D perlin;

float3 getCartesianDirection(Rays s) {
  float st = sin(s.theta);
  float ct = cos(s.theta);
  float sp = sin(s.phi);
  float cp = cos(s.phi);

  // Standard spherical basis vectors in Cartesian space
  float3 e_r = float3(st * cp, st * sp, ct);
  float3 e_theta = float3(ct * cp, ct * sp, -st);
  float3 e_phi = float3(-sp, cp, 0.0f);

  float st_safe = (fabs(st) < 1e-4f) ? (st < 0.0f ? -1e-4f : 1e-4f) : st;
  float3 dir =
      s.dr * e_r + s.r * s.dtheta * e_theta + s.r * st_safe * s.dphi * e_phi;
  return normalize(dir);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compute entry point — ported from raytrace.cu pixelProcess()
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 id: SV_DispatchThreadID) {

  int x = int(id.x);
  int y = int(id.y);

  int WIDTH = params.width;
  int HEIGHT = params.height;

  // Bounds check
  if (x >= WIDTH || y >= HEIGHT)
    return;

  // DEBUG: Test that compute shader is running
  // Uncomment the next line to verify the compute pipeline works:
  // outputImage[int2(x, y)] = float4(1.0, 0.0, 0.0, 1.0);
  // return;

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // unpacking (temporário)
  // @{

  float3 pos    = float3(params.pos);
  float3 right  = float3(params.right);
  float3 fwd    = float3(params.fwd);
  float3 up     = float3(params.up);
  float fov_y   = float(params.fov_y);
  float rs      = float(RS);

  // @}

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // posição da câmera em cartesianos
  // @{
  /*
      alocando u,v, as coordenadas da tela.

          → relaciona x,y à aspect_ratio e u,v.
          → fov_y controla quantos graus temos visão.

  */

  float aspect = float(WIDTH) / float(HEIGHT);
  float tanHalfFov = tan(fov_y * 0.5f * float(PI) / 180.0f);
  float u =
      (2.0f * (float(x) + 0.5f) / float(WIDTH) - 1.0f) * aspect * tanHalfFov;
  float v = (1.0f - 2.0f * (float(y) + 0.5f) / float(HEIGHT)) * tanHalfFov;

  /*
      determinando a direção do raio:

          ◦ cálculo de onde apontam na tela: combinação linear,
          ◦ se relacionam à [fwd, right, up]:
              → fwd: centro da imagem (profundidade),
              → right: deslocamento horizontal (u),
              → up: deslocamento vertical (v).
          ◦ normalização,
          ◦ usa-se float para prezar pela precisão.
  */

  // direção do raio: u*right + v*up + forward, normalizada
  float dx = u * right.x + v * up.x + fwd.x;
  float dy = u * right.y + v * up.y + fwd.y;
  float dz = u * right.z + v * up.z + fwd.z;
  float dlen = sqrt(dx * dx + dy * dy + dz * dz);
  dx /= dlen;
  dy /= dlen;
  dz /= dlen;

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // transformando para coordenadas esféricas e inicializando Rays

  /*
      cartesianas → esféricas:

          ◦ por mais incrível que pareça, pelo fato de que Schwarzschild é
            modelado em esféricas, as equações são mais fáceis do que em
     cartesianas.

          ◦ relembrando:
              → r     = distância radial ao centro do buraco negro
              → theta = ângulo polar (0 = polo norte, π/2 = equador, π = polo
     sul) → phi   = ângulo azimutal (rotação em torno do eixo z)
  */

  float ox = pos.x;
  float oy = pos.y;
  float oz = pos.z;

  float r0 = sqrt(ox * ox + oy * oy + oz * oz);
  float theta0 = acos(clamp(oz / r0, -1.0f, 1.0f));
  float phi0 = atan2(oy, ox);

  if (theta0 > float(PI) - 1e-6f)
    theta0 = float(PI) - 1e-6f;
  if (theta0 < 1e-6f)
    theta0 = 1e-6f;

  /*
      • coordenadas cartesianas → esféricas via Jacobiano da transformação
     esférica → cartesiana → x = r * sin(θ) * cos(φ) → y = r * sin(θ) * sin(φ)
          → z = r * cos(θ)

          ◦   Derivando cada componente em relação a (r, θ, φ) e invertendo,
     obtemos como (dx, dy, dz) se traduz em (dr, dθ, dφ).

          ◦ st_safe protege dphi contra divisão por zero quando θ = 0 ou π
     (polos).
  */

  float st = sin(theta0), ct = cos(theta0), sp = sin(phi0), cp = cos(phi0);
  float st_safe = (fabs(st) < 1e-4f) ? (st < 0.0f ? -1e-4f : 1e-4f) : st;

  /*
      E = energia por unidade de massa (conservada porque a métrica não depende
     de t) L = momento angular por unidade de massa (conservada porque não
     depende de φ) ↓ são ctes, usadas a cada step do RK4 para calcular dt/dλ =
     E/f sem precisar rastrear o tempo coordenado t


      f0 = fator de Schwarzschild na posição inicial da câmera.
      vmag é a magnitude da velocidade 3D no espaço de Schwarzschild.
  */

  Rays s;
  s.r = r0;
  s.theta = theta0;
  s.phi = phi0;
  s.dr = st * cp * dx + st * sp * dy + ct * dz;
  s.dtheta = (ct * cp * dx + ct * sp * dy - st * dz) / r0;
  s.dphi = (-sp * dx + cp * dy) / (r0 * st_safe);

  if (fabs(st) < 1e-4f) {
    s.dphi = 0.0f;
  }

  float f0 = 1.0f - rs / r0;
  float vmag = sqrt(s.dr * s.dr / f0 + r0 * r0 * s.dtheta * s.dtheta +
                    r0 * r0 * st_safe * st_safe * s.dphi * s.dphi);

  s.L = r0 * r0 * st * st * s.dphi;
  s.E = f0 * vmag;
  s.is_trapped = false;

  if (!isfinite(s.r) || !isfinite(s.phi) || !isfinite(s.theta)) {
    outputImage[int2(x, y)] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // @}

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // parâmetros importantes para a simulação.
  // @{

  const float step = rs * float(STEP_FACTOR);
  const float escape_radius = rs * float(ESCAPE_FACTOR);

  const float disk_r1 = rs * 3.0f;
  const float disk_r2 = rs * 12.0f;

  float accum_r = 0.0f; // canal vermelho acumulado
  float accum_g = 0.0f; // canal verde acumulado
  float accum_b = 0.0f; // canal azul acumulado
  float accum_alpha =
      0.0f; // opacidade acumulada [0,1]
            // quando accum_alpha ≥ 1.0 → disco completamente opaco → break

  float y_prev = s.r * cos(s.theta);
  float y_next = y_prev;

  RayResult result = RayResult::NONE;

  float bg_r = 0.0f;
  float bg_g = 0.0f;
  float bg_b = 0.0f;

  // @}

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // loop em si
  // @{

  // ─────────────────────────────────────────────────────────────────────────────────────────────────
  // escape por parâmetro de impacto:
  // @{
  /*

      •   uma forma de Early Exit que analisa a relação entre o impacto da
     energia de um raio para determinar se ele deve ser integrado. A lógica é:
     se um raio tiver energia e momento angular, quando relacionados por 'b',
     maior que um threshold predeterminado como LIMITE que um raio integrado
     tem, ele deve ser cortado.

  */

  {
    float b = (s.E > 1e-10f) ? fabs(s.L / s.E) : 1e30f;
    const float b_crit = 2.598f * rs; // 3√3/2 * rs

    if (b > b_crit * float(IMPACT_CUTOFF) && s.r > rs * 3.0f && s.dr > 0.0f) {
      // s.phi = fmod(s.phi, 2.0f * float(PI));
      // if(s.phi >  float(PI)) s.phi -= 2.0f * float(PI);
      // if(s.phi < -float(PI)) s.phi += 2.0f * float(PI);

      float3 dir = getCartesianDirection(s);
      float phi_bg = atan2(dir.y, dir.x);
      float theta_bg = acos(clamp(dir.z, -1.0f, 1.0f));

      // raio escapa com certeza — amostra starmap direto
      float u_tex = phi_bg / (2.0f * float(PI)) + 0.5f;
      float v_tex = theta_bg / float(PI);
      float4 color = starmap.SampleLevel(float2(u_tex, v_tex), 0.0);

      bg_r = color.x;
      bg_g = color.y;
      bg_b = color.z;
      result = RayResult::ESCAPE;

      // Write and return early
      outputImage[int2(x, y)] = float4(bg_r, bg_g, bg_b, 1.0);
      return;
    }
  }
  // @}

  for (int i = 0; i < MAX_STEPS; i++) {

    float r_prev = s.r;
    float phi_prev = s.phi;
    float dphi_prev = s.dphi;
    float theta_prev = s.theta;

    // testar o tempo disso aqui
    if (s.r > disk_r2 * 1.2f && s.dr > 0.0f && i > 50) {
      // escapa
      break;
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // horizonte pré-step
    if (s.is_trapped || s.r <= rs || s.r <= 0.0f) {
      s.is_trapped = true;
      bg_r = bg_g = bg_b = 0.0f;
      result = RayResult::HORIZON;
      break;
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // passo adaptativo
    // @{
    /*
        •   perto do horizonte (r < 5*limits), a curvatura do espaço-tempo
       cresce rapidamente. um step fixo grande causaria que o raio "pulasse" o
       horizonte — de r=1.05*rs para r=0.5*rs num único passo, sem o teste de
       captura disparar.

            o adaptive_step reduz linearmente com r: a 1*rs o passo é 1/5 do
       base_step. o mínimo de adaptive_clamp * step evita que o loop fique
       infinitamente lento para raios que raspam o horizonte.
    */

    float adaptive_step = step;
    if (s.r < float(ADAPTIVE_FACTOR) * rs) {
      adaptive_step =
          step *
          (s.r /
           (rs * float(ADAPTIVE_FACTOR))); // escala linear: a 1rs → step/5

      if (adaptive_step < step * float(ADAPTIVE_CLAMP))
        adaptive_step = step * float(ADAPTIVE_CLAMP); // mínimo
    }

    float sin_theta_now = fabs(sin(s.theta));
    if (sin_theta_now < 0.05f) {
      float pole_scale = max(sin_theta_now / 0.05f, float(ADAPTIVE_CLAMP));
      adaptive_step *= pole_scale;
    }
    // @}

    // integrador rk4
    rk4Step(s, adaptive_step, rs);

    // ── Polar reflection ─────────────────────────────────────────────
    // Spherical coords require theta ∈ [0, PI]. If the integrator steps
    // through a pole, reflect theta and rotate phi by π.
    if (s.theta < 0.0f) {
      s.theta = -s.theta;
      s.phi += float(PI);
      s.dtheta = -s.dtheta;
    }
    if (s.theta > float(PI)) {
      s.theta = 2.0f * float(PI) - s.theta;
      s.phi += float(PI);
      s.dtheta = -s.dtheta;
    }
    // Keep phi in [-π, π]
    if (s.phi > float(PI))
      s.phi -= 2.0f * float(PI);
    if (s.phi < -float(PI))
      s.phi += 2.0f * float(PI);
    // ─────────────────────────────────────────────────────────────────

    // Explosion guard — if the step exploded, restore previous state
    if (s.r > r_prev * 100.0f || fabs(s.theta) > 10.0f * float(PI)) {
      s.r = r_prev;
      s.theta = theta_prev;
      s.phi = phi_prev;
      s.dtheta = 0.0f;
      s.dphi = dphi_prev;
    }

    // Terminate loop early if step marked the ray as trapped
    if (s.is_trapped) {
      bg_r = bg_g = bg_b = 0.0f;
      result = RayResult::HORIZON;
      break;
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // escape pré-step
    // @{
    /*
        quando o raio escapa (r > escape_radius), o ângulo final (θ, φ)
       representa a direção de onde a luz vem — que é a direção em que a câmera
       "vê" esse pixel.

        converte (theta, phi) do ponto de fuga para UV do mapa equiretangular
            → theta: [0, π]    → V: [0, 1]   (polo norte = 0, polo sul = 1)
            → phi:   [-π, π]   → U: [0, 1]   (wrapa em 360°)

        O lensing gravitacional dobra os raios, então estrelas que "deveriam"
       estar numa direção aparecem deslocadas — é exatamente o efeito visual de
       lensing.
    */

    if (s.r > escape_radius) {
      s.phi = fmod(s.phi, 2.0f * float(PI));
      if (s.phi > float(PI))
        s.phi -= 2.0f * float(PI);
      if (s.phi < -float(PI))
        s.phi += 2.0f * float(PI);

      float u_tex = s.phi / (2.0f * float(PI)) + 0.5f; // [-π,π] → [0,1]
      float v_tex = s.theta / float(PI);               // [0,π]  → [0,1]

      float4 color = starmap.SampleLevel(float2(u_tex, v_tex), 0.0);

      bg_r = color.x;
      bg_g = color.y;
      bg_b = color.z;

      result = RayResult::ESCAPE;

      break;
    }
    // @}

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // horizonte pós-step
    if (s.r <= rs || s.r <= 0.0f || !isfinite(s.r)) {
      s.is_trapped = true;
      bg_r = bg_g = bg_b = 0.0f;
      result = RayResult::HORIZON;

      break;
    }
    y_next = s.r * cos(s.theta);

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // early exit por MAX_STEPS parcial
    // @{
    /*
        • se o raio já deu muitos steps, está longe do BH e se afastando →
       escapa
    */

    if (i > MAX_STEPS / MAX_STEPS_DIV && s.r > disk_r2 * 1.5f && s.dr > 0.0f) {
      // s.phi = fmod(s.phi, 2.0f * float(PI));
      // if (s.phi > float(PI))
      //   s.phi -= 2.0f * float(PI);
      // if (s.phi < -float(PI))
      //   s.phi += 2.0f * float(PI);

      float3 dir = getCartesianDirection(s);
      float phi_bg = atan2(dir.y, dir.x);
      float theta_bg = acos(clamp(dir.z, -1.0f, 1.0f));

      float u_tex = phi_bg / (2.0f * float(PI)) + 0.5f;
      float v_tex = theta_bg / float(PI);
      float4 color = starmap.SampleLevel(float2(u_tex, v_tex), 0.0);

      bg_r = color.x;
      bg_g = color.y;
      bg_b = color.z;

      result = RayResult::ESCAPE;

      break;
    }
    // @}

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // case: disco
    // @{

    // detecção por emissividade de disco volumétrico
    float x_cartesiano = s.r * sin(s.theta) * cos(s.phi);
    float y_cartesiano = s.r * sin(s.theta) * sin(s.phi);
    float z_cartesiano = s.r * cos(s.theta);

    float r_current =
        sqrt(x_cartesiano * x_cartesiano + y_cartesiano * y_cartesiano);
    float emissividade = diskEmissivity(r_current, z_cartesiano, disk_r1,
                                        disk_r2, float(DISK_HEIGHT_SCALE));

    if (emissividade > float(EMISSIVITY_RATE)) {

      /*

          Aqui usamos a composição de 3 efeitos físicos:

              → Perlin noise:     modula o brilho local — gás turbulento, não
         uniforme → Doppler shift:    assimetria orbital — lado se aproximando
         brilha mais → Redshift grav:    raios que saem de r pequeno perdem
         energia


          • O beaming concentra a emissão na direção do movimento:

              lado se aproximando → feixe comprimido → mais brilhante
              lado se afastando   → feixe dilatado   → mais escuro


          • Separamos os dois para controle visual independente:

              freq_effect  = doppler * rs_freq   → muda a cor
              beam_effect  = doppler^3            → muda o brilho


          O expoente 3 vem de:
              → 1 fator de energia do fóton
              → fatores de aberração relativística (compressão do feixe)

              ◦ Fisicamente seria D^4 mas D^3 dá resultado visualmente mais
         equilibrado.

      */

      float doppler = dopplerShift(s.phi, r_current, float3(pos), rs);
      float rs_freq = redShift(r_current, rs);

      float perlin_noise =
          perlinNoise(perlin, r_current, s.phi, disk_r1, disk_r2);
      float base_brightness = 0.4f + 0.6f * perlin_noise;

      float freq_effect = doppler * rs_freq;
      float beam_effect = pow(max(doppler, 0.01f), 3.0f);

      float temp = (r_current - disk_r1) / (disk_r2 - disk_r1);
      float temp_norm = pow(1.0f - temp, 0.75f);

      float disk_r, disk_g, disk_b;
      temperatureToColor(temp_norm, disk_r, disk_g, disk_b);

      // modula a cor base pelo shift de frequência
      // freq > 1 → empurra para branco/azul (se aproximando)
      // freq < 1 → empurra para vermelho (se afastando)
      if (freq_effect > 1.0f) {
        float blend = min((1.0f - freq_effect) / 1.5f, 1.0f);
        disk_g = disk_g + blend * (1.0f - disk_g);
        disk_b = disk_b + blend * (1.0f - disk_b);
      } else {
        float blend = min((1.0f - freq_effect) / 0.8f, 1.0f);
        disk_r = disk_r + blend * 0.3f;
        disk_g = disk_g + blend * (1.0f - disk_g);
        disk_b = disk_b + blend * (1.0f - disk_b);
      }

      float intensity =
          emissividade * base_brightness * float(EMISSION_SCALE) * beam_effect;

      float step_normalized = (adaptive_step / (disk_r2 - disk_r1));
      float contribution = intensity * step_normalized * float(DISK_OPACITY);
      float remaining = 1.0f - accum_alpha;

      accum_r += disk_r * contribution * remaining;
      accum_g += disk_g * contribution * remaining;
      accum_b += disk_b * contribution * remaining;

      accum_alpha += contribution * remaining;

      result = RayResult::DISK;

      // disco completamente opaco → para de integrar
      if (accum_alpha >= 1.0f) {
        accum_alpha = 1.0f;
        result = RayResult::DISK;
        break;
      }

      // @}
    }

    y_prev = y_next;
  }
  // @}

  // ─────────────────────────────────────────────────────────────────────────────────────────────────
  // fallback:
  // @{
  /*
      Após o loop, bg_r=bg_g=bg_b=0 significa que nenhum dos casos anteriores
     disparou. Dois subcasos: → s.r < rs * closeness  → raio ficou preso perto
     do horizonte → preto (capturado) → s.r ≥ rs * closeness  → raio ficou
     orbitando longe → amostra starmap na direção atual

          • A escrita final:
              ◦ int idx = (y * WIDTH + x) * 3;
              ◦ pixels[idx+0] = R;   // byte 0 do pixel (x,y) = canal R
              ◦ pixels[idx+1] = G;   // byte 1 = canal G
              ◦ pixels[idx+2] = B;   // byte 2 = canal B

          • engine.cpp faz upload disso direto para a textura OpenGL.
  */

  if (result == RayResult::NONE) {
    result = RayResult::FALLBACK;
  }

  if (result == RayResult::HORIZON) {
    bg_r = bg_g = bg_b = 0.0f;

  } else if (result == RayResult::ESCAPE || result == RayResult::FALLBACK) {

    if (result == RayResult::FALLBACK && s.r < rs * float(CLOSENESS)) {
      bg_r = bg_g = bg_b = 0.0f;

    } else {

      float3 dir = getCartesianDirection(s);
      float phi_bg = atan2(dir.y, dir.x);
      float theta_bg = acos(clamp(dir.z, -1.0f, 1.0f));
      float u_tex = frac(phi_bg / (2.0f * float(PI)) + 0.5f);
      float v_tex = clamp(theta_bg / float(PI), 0.0f, 1.0f);
      float4 color = starmap.SampleLevel(float2(u_tex, v_tex), 0.0);

      bg_r = color.x;
      bg_g = color.y;
      bg_b = color.z;
    }
  }

  // @}

  // ─────────────────────────────────────────────────────────────────────────────────────────────────
  // Final compositing: disk + background
  // ─────────────────────────────────────────────────────────────────────────────────────────────────

  float out_r, out_g, out_b;

  if (accum_alpha > 0.0001f) {
    float bg_weight = 1.0f - min(accum_alpha, 1.0f);
    out_r = accum_r + bg_r * bg_weight;
    out_g = accum_g + bg_g * bg_weight;
    out_b = accum_b + bg_b * bg_weight;

    // relativos à sensibilidade de cada cor no olho humano
    float luminance = 0.299f * out_r + 0.587f * out_g + 0.114f * out_b;

    if (luminance < 0.01f && result == RayResult::DISK) {
      float u_tex = s.phi / (2.0f * float(PI)) + 0.5f;
      float v_tex = s.theta / float(PI);
      float4 color = starmap.SampleLevel(float2(u_tex, v_tex), 0.0);
      float t = 0.6f;
      out_r = out_r * (1.0f - t) + color.x * t;
      out_g = out_g * (1.0f - t) + color.y * t;
      out_b = out_b * (1.0f - t) + color.z * t;
    }

  } else {
    out_r = bg_r;
    out_g = bg_g;
    out_b = bg_b;
  }

  outputImage[int2(x, y)] =
      float4(min(out_r, 1.0f), min(out_g, 1.0f), min(out_b, 1.0f), 1.0f);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
