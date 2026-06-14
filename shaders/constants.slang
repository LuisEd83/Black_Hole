#pragma once

// ========= PHYSICS RELATED PARAMETERS ===============
static const double PI = 3.14159265358979323846;

// Speed of Light (m/s)
static const double c = 299792458.0;

// Universal Gravitational Constant (m³/(kg·s²))
static const double G = 6.67430e-11;

// Sagittarius A* mass (kg)
static const double BH_MASS = 8.54e36;

// Schwarzschild radius: r_s = 2GM/c²
// Event horizon radius (within this, light doesn't escape)
static const double RS_PHYS = 2.0 * G * BH_MASS / (c * c);
// In-shader RS is normalized to 1.0 — all distances are in units of
// Schwarzschild radius. This improves float precision by ~10x.
static const double RS = 1.0;

enum class RayResult { NONE, HORIZON, ESCAPE, DISK, FALLBACK };

// ========= PARAMETER STRUCTS ===============
struct RenderParams {
  int width;
  int height;
  double3 pos;
  double3 fwd;
  double3 right;
  double3 up;
  float fov_y;
};

// ========= SIMULATION PARAMETERS ===============

static const double CAMERA_FACTOR = 12.0;
static const double FOV_Y = 60.0;

static const double X_COEF = 1.0;
static const double Y_COEF = 1.1;
static const double Z_COEF = 0.7;

static const float x_multiplier = 0.0f;
static const float y_multiplier = 0.0f;
static const float z_multiplier = 0.0f;

static const float x_bias = 0.0f;
static const float y_bias = 0.0f;
static const float z_bias = 0.0f;

static const int MAX_STEPS = 10000;
static const double STEP_FACTOR = 0.1;

static const double ESCAPE_FACTOR = 200.0;

// ========= ADVANCED SIMULATION PARAMETERS ===============
static const double IMPACT_CUTOFF = 7.5;
static const double MAX_STEPS_DIV = 2.5;

static const double DISK_HEIGHT_SCALE = 0.025;
static const double DISK_OPACITY = 3.0;
static const double EMISSION_SCALE = 2.5;

static const double ADAPTIVE_CLAMP = 0.01;
static const double CLOSENESS = 5.0;

static const double ADAPTIVE_FACTOR = 5.0;
static const double EMISSIVITY_RATE = 0.01;
