// Component formulas with derivatives, in C++: the native counterpart of the dual-number
// evaluation in differometor/fast.py (`Dual`, `_NumpyShim`, `_EntryBlock.run`).
//
// What it replaces
//   fast.py evaluates the component functions of differometor/components.py on dual numbers
//   built from NumPy arrays: every arithmetic operation of a formula is one or more NumPy calls
//   on small arrays. Here one call evaluates a whole group of calls of one component function,
//   element by element, with the same dual-number rules.
//
// What stays the same
//   The formulas are the ones of components.py, written out again below with the same
//   operations in the same order (C++ operators on the type `D`). The rules for derivatives
//   are those of fast.py: product and quotient rule, `_unary` for exp/sin/cos/sinh/cosh/radians,
//   `_chain` for sqrt and arcsin (a direction whose tangent is exactly 0 keeps the derivative 0
//   where the slope is infinite), real functions for real values (sqrt of a negative real
//   number and arcsin outside [-1, 1] are nan, as in NumPy).
//
// Shapes
//   fast.py lets broadcasting decide whether a value or a derivative has one column or one per
//   row of the sweep, and whether an output has a derivative at all. `D` carries the same three
//   facts as flags (`active`, `vs`, `ts`) through every operation, and `ent_eval` reports them
//   per output, so that the caller hands on arrays of the shapes the NumPy path produces.
//
// components.py stays the reference: differometor/_native/test_entries.py compares every
// function here with the NumPy evaluation on random inputs.
#include <cmath>
#include <complex>
#include <cstring>

typedef std::complex<double> cplx;

namespace {

const int MAXW = 7;  // a component function has 7 inputs, so at most 7 derivative directions
int W = 0;           // directions of the current call (set by ent_eval; callers hold the GIL)

// Constants of components.py.
const double EPSILON_0C = 1.0;
const double C_LIGHT = 299792458.0;
const double LAMBDA = 1.064e-06;
const double UNIT_VACUUM = 1.0;
const double F0 = C_LIGHT / LAMBDA;
const double X_SCALE = 1e-09;
const double PI = 3.141592653589793;
const double E = 2.718281828459045;
const cplx I(0.0, 1.0);

struct D {
    cplx v;
    cplx t[MAXW];
    bool active;  // has a derivative (a `Dual` in fast.py; otherwise a plain value)
    bool real;    // the value is real-typed (decides between the real and the complex function)
    bool vs;      // the value differs along the sweep
    bool ts;      // the derivative differs along the sweep

    D() : v(0.0), active(false), real(true), vs(false), ts(false) {}
    D(double x) : v(x), active(false), real(true), vs(false), ts(false) {}
    D(cplx x) : v(x), active(false), real(false), vs(false), ts(false) {}
};

inline D operator-(const D& a) {
    D r = a;
    r.v = -a.v;
    if (a.active)
        for (int i = 0; i < W; i++) r.t[i] = -a.t[i];
    return r;
}

inline D operator+(const D& a, const D& b) {
    D r;
    r.v = a.v + b.v;
    r.real = a.real && b.real;
    r.vs = a.vs || b.vs;
    r.active = a.active || b.active;
    if (a.active && b.active) {
        for (int i = 0; i < W; i++) r.t[i] = a.t[i] + b.t[i];
        r.ts = a.ts || b.ts;
    } else if (a.active) {
        for (int i = 0; i < W; i++) r.t[i] = a.t[i];
        r.ts = a.ts;
    } else if (b.active) {
        for (int i = 0; i < W; i++) r.t[i] = b.t[i];
        r.ts = b.ts;
    }
    return r;
}

inline D operator-(const D& a, const D& b) {
    D r;
    r.v = a.v - b.v;
    r.real = a.real && b.real;
    r.vs = a.vs || b.vs;
    r.active = a.active || b.active;
    if (a.active && b.active) {
        for (int i = 0; i < W; i++) r.t[i] = a.t[i] - b.t[i];
        r.ts = a.ts || b.ts;
    } else if (a.active) {
        for (int i = 0; i < W; i++) r.t[i] = a.t[i];
        r.ts = a.ts;
    } else if (b.active) {
        for (int i = 0; i < W; i++) r.t[i] = -b.t[i];
        r.ts = b.ts;
    }
    return r;
}

inline D operator*(const D& a, const D& b) {
    D r;
    r.v = a.v * b.v;
    r.real = a.real && b.real;
    r.vs = a.vs || b.vs;
    r.active = a.active || b.active;
    if (a.active && b.active) {  // self.tan * other.val + self.val * other.tan
        for (int i = 0; i < W; i++) r.t[i] = a.t[i] * b.v + a.v * b.t[i];
        r.ts = a.ts || b.vs || a.vs || b.ts;
    } else if (a.active) {  // self.tan * other
        for (int i = 0; i < W; i++) r.t[i] = a.t[i] * b.v;
        r.ts = a.ts || b.vs;
    } else if (b.active) {
        for (int i = 0; i < W; i++) r.t[i] = b.t[i] * a.v;
        r.ts = b.ts || a.vs;
    }
    return r;
}

inline D operator/(const D& a, const D& b) {
    D r;
    r.v = a.v / b.v;
    r.real = a.real && b.real;
    r.vs = a.vs || b.vs;
    r.active = a.active || b.active;
    if (a.active && b.active) {  // (self.tan - quotient * other.tan) / other.val
        for (int i = 0; i < W; i++) r.t[i] = (a.t[i] - r.v * b.t[i]) / b.v;
        r.ts = a.ts || b.ts || a.vs || b.vs;
    } else if (a.active) {  // self.tan / other
        for (int i = 0; i < W; i++) r.t[i] = a.t[i] / b.v;
        r.ts = a.ts || b.vs;
    } else if (b.active) {  // -(quotient * self.tan) / self.val
        for (int i = 0; i < W; i++) r.t[i] = -(r.v * b.t[i]) / b.v;
        r.ts = a.vs || b.vs || b.ts;
    }
    return r;
}

// x ** 2: NumPy squares; the derivative is `exponent * val ** (exponent - 1) * tan`.
inline D square(const D& a) {
    D r;
    r.v = a.v * a.v;
    r.real = a.real;
    r.vs = a.vs;
    r.active = a.active;
    if (a.active) {
        for (int i = 0; i < W; i++) r.t[i] = (2.0 * a.v) * a.t[i];
        r.ts = a.vs || a.ts;
    }
    return r;
}

// A real slope times a real tangent stays real (inf * 0 in an imaginary part would be nan).
inline cplx scaled(bool real, cplx slope, cplx tangent) {
    if (real && slope.imag() == 0.0 && tangent.imag() == 0.0) return cplx(slope.real() * tangent.real(), 0.0);
    return slope * tangent;
}

// `_unary(function, derivative)`: tan = derivative(x, y) * x.tan, the slope has the shape of the value.
inline D with_slope(const D& a, cplx y, cplx slope) {
    D r;
    r.v = y;
    r.real = a.real;
    r.vs = a.vs;
    r.active = a.active;
    if (a.active) {
        for (int i = 0; i < W; i++) r.t[i] = scaled(a.real, slope, a.t[i]);
        r.ts = a.vs || a.ts;
    }
    return r;
}

// `_chain`: where the slope is not finite, a direction with tangent exactly 0 keeps 0.
inline D with_singular_slope(const D& a, cplx y, cplx slope) {
    D r = with_slope(a, y, slope);
    if (a.active && !(std::isfinite(slope.real()) && std::isfinite(slope.imag())))
        for (int i = 0; i < W; i++)
            if (a.t[i] == cplx(0.0)) r.t[i] = 0.0;
    return r;
}

inline D exp(const D& a) {
    cplx y = a.real ? cplx(std::exp(a.v.real())) : std::exp(a.v);
    return with_slope(a, y, y);
}

inline D sin(const D& a) {
    if (a.real) return with_slope(a, std::sin(a.v.real()), std::cos(a.v.real()));
    return with_slope(a, std::sin(a.v), std::cos(a.v));
}

inline D cos(const D& a) {
    if (a.real) return with_slope(a, std::cos(a.v.real()), -std::sin(a.v.real()));
    return with_slope(a, std::cos(a.v), -std::sin(a.v));
}

inline D sinh(const D& a) {
    if (a.real) return with_slope(a, std::sinh(a.v.real()), std::cosh(a.v.real()));
    return with_slope(a, std::sinh(a.v), std::cosh(a.v));
}

inline D cosh(const D& a) {
    if (a.real) return with_slope(a, std::cosh(a.v.real()), std::sinh(a.v.real()));
    return with_slope(a, std::cosh(a.v), std::sinh(a.v));
}

inline D sqrt(const D& a) {
    if (a.real) {
        double y = std::sqrt(a.v.real());
        return with_singular_slope(a, y, 0.5 / y);
    }
    cplx y = std::sqrt(a.v);
    return with_singular_slope(a, y, 0.5 / y);
}

inline D arcsin(const D& a) {
    if (a.real) {
        double x = a.v.real();
        return with_singular_slope(a, std::asin(x), 1.0 / std::sqrt(1.0 - x * x));
    }
    return with_singular_slope(a, std::asin(a.v), 1.0 / std::sqrt(1.0 - a.v * a.v));
}

// np.radians(x) is x * (pi / 180); its derivative is a constant, so `ts` is unchanged.
inline D radians(const D& a) {
    const double factor = PI / 180.0;
    D r = a;
    r.v = a.v * factor;
    if (a.active)
        for (int i = 0; i < W; i++) r.t[i] = factor * a.t[i];
    return r;
}

inline D conj(const D& a) {
    D r = a;
    r.v = std::conj(a.v);
    if (a.active)
        for (int i = 0; i < W; i++) r.t[i] = std::conj(a.t[i]);
    return r;
}

// ---------------------------------------------------------------------------------------
// The formulas of components.py. Same operations, same order.
// ---------------------------------------------------------------------------------------

inline D phase_shift(const D& tuning, const D& frequency, const D& refractive_index) {
    return D(2.0) * radians(tuning) * refractive_index * (D(1.0) + frequency / D(F0));
}

inline D incidence_angle_left(const D& alpha) { return cos(radians(alpha)); }

inline D incidence_angle_right(const D& alpha, const D& index_left, const D& index_right) {
    return cos(arcsin(index_left / index_right * sin(radians(alpha))));
}

inline D reflectivity_left(const D& reflectivity, const D& tuning, const D& frequency, const D& index_left, const D& alpha) {
    return -sqrt(reflectivity) * exp(D(I) * phase_shift(tuning, frequency, index_left) * incidence_angle_left(alpha));
}

inline D reflectivity_right(const D& reflectivity, const D& tuning, const D& frequency, const D& index_left, const D& index_right, const D& alpha) {
    return -sqrt(reflectivity) * exp(D(-I) * phase_shift(tuning, frequency, index_right) * incidence_angle_right(alpha, index_right, index_left));
}

inline D transmissivity(const D& value, const D& tuning, const D& frequency, const D& index_left, const D& index_right, const D& alpha) {
    D left_term = phase_shift(tuning, frequency, index_left) * incidence_angle_left(alpha);
    D right_term = phase_shift(tuning, frequency, index_right) * incidence_angle_right(alpha, index_left, index_right);
    return -sqrt(value) * exp(D(I) * (D(PI / 2) + D(0.5) * (left_term - right_term)));
}

inline D optomechanical_phase_left(const D& alpha) {
    return D(I * X_SCALE * 2.0 * PI / LAMBDA) * incidence_angle_left(alpha);
}

inline D optomechanical_phase_right(const D& alpha, const D& index_left, const D& index_right) {
    return D(I * X_SCALE * 2.0 * PI / LAMBDA) * incidence_angle_right(alpha, index_left, index_right);
}

inline D tuning_correction(const D* p) { return exp(D(I) * radians(p[1]) * p[0] / D(F0)); }

// Every function writes four outputs, like `standardize_output` (missing ones are 0).

void space_at(const D& p0, const D& p1, const D& p2, D* out) {
    out[0] = -exp(D(-I * 2.0 * PI) * p0 * p1 * p2 / D(C_LIGHT));
}

void space(const D* p, D* out) { space_at(p[0], p[1], p[2], out); }

void space_lower(const D* p, D* out) { space_at(-p[0], p[1], p[2], out); }

void laser(const D* p, D* out) {
    out[0] = sqrt(D(2.0) * p[0] / D(EPSILON_0C)) * exp(D(I) * radians(p[1]));
}

void squeezer(const D* p, D* out) {
    const double vacuum_unit = UNIT_VACUUM / 2;
    D squeezing_parameter = p[0] / D(20 * std::log10(E));
    D phase = exp(D(I * 2.0) * radians(p[1]));
    out[0] = D(vacuum_unit) * cosh(D(2.0) * squeezing_parameter);
    out[1] = D(vacuum_unit) * sinh(D(2.0) * squeezing_parameter) * phase;
    out[2] = conj(out[0]);
    out[3] = conj(out[1]);
}

void signal_function(const D* p, D* out) { out[0] = p[0] * exp(D(I) * radians(p[1])); }

void laser_amplitude_modulation(const D*, D* out) { out[0] = D(EPSILON_0C * 0.5); }

void laser_frequency_modulation(const D* p, D* out) { out[0] = D(EPSILON_0C * 0.5) / p[0]; }

void laser_frequency_modulation_lower(const D* p, D* out) { out[0] = D(EPSILON_0C * 0.5) / (-p[0]); }

void space_modulation_at(const D& p0, const D& p1, const D& p2, D* out) {
    D w_g = D(2 * PI) * p0;
    const double w_0 = 2 * PI * F0;
    D m_g = D(-0.5 * w_0) / w_g * sin(w_g * p1 * p2 / D(2.0) / D(C_LIGHT));
    D phi_sb = -w_g * p1 * p2 / D(2.0) / D(C_LIGHT);
    out[0] = D(I) * m_g * exp(D(I) * phi_sb);
}

void space_modulation(const D* p, D* out) { space_modulation_at(p[0], p[1], p[2], out); }

void space_modulation_lower(const D* p, D* out) { space_modulation_at(-p[0], p[1], p[2], out); }

void negate_all(D* out) {
    for (int o = 0; o < 4; o++) out[o] = -out[o];  // `-standardize_output(...)`: the padding becomes -0.0
}

void susceptibility(const D* p, D* out) {
    out[0] = D(-1.0) / (p[1] * square(D(2 * PI) * p[0]));
    negate_all(out);
}

void force_calculation_left(const D* p, D* out) {
    out[0] = incidence_angle_left(p[0]) / D(C_LIGHT * X_SCALE);
    negate_all(out);
}

void force_calculation_right(const D* p, D* out) {
    out[0] = incidence_angle_right(p[0], p[1], p[2]) / D(C_LIGHT * X_SCALE);
}

void corrected_optomechanical_phase_left(const D* p, D* out) {
    D absolute_reflectivity = (D(1.0) - p[3]) * p[2];
    out[0] = tuning_correction(p) * optomechanical_phase_left(p[5]) * reflectivity_left(absolute_reflectivity, p[1], p[0], p[4], p[5]);
    negate_all(out);
}

void corrected_optomechanical_phase_right(const D* p, D* out) {
    D absolute_reflectivity = (D(1.0) - p[3]) * p[2];
    out[0] = conj(tuning_correction(p)) * D(-1.0) * optomechanical_phase_right(p[6], p[4], p[5]) *
             reflectivity_right(absolute_reflectivity, p[1], p[0], p[4], p[5], p[6]);
    negate_all(out);
}

void vacuum_quantum_noise(const D*, D* out) { out[0] = D(UNIT_VACUUM / 2); }

void loss_quantum_noise(const D* p, D* out) { out[0] = p[0] / D(2.0); }

void surface(const D* p, D* out) {
    D absolute_reflectivity = (D(1.0) - p[0]) * p[1];
    D absolute_transmissivity = (D(1.0) - p[0]) * (D(1.0) - p[1]);
    out[0] = reflectivity_left(absolute_reflectivity, p[2], p[3], p[4], p[6]);
    out[1] = transmissivity(absolute_transmissivity, p[2], p[3], p[4], p[5], p[6]);
    out[2] = reflectivity_right(absolute_reflectivity, p[2], p[3], p[4], p[5], p[6]);
}

void dummy_function(const D*, D* out) { out[0] = D(1.0); }

struct Entry {
    const char* name;
    void (*function)(const D*, D*);
};

const Entry TABLE[] = {
    {"force_calculation_right", force_calculation_right},
    {"laser", laser},
    {"surface", surface},
    {"space", space},
    {"space_modulation", space_modulation},
    {"signal_function", signal_function},
    {"dummy_function", dummy_function},
    {"vacuum_quantum_noise", vacuum_quantum_noise},
    {"loss_quantum_noise", loss_quantum_noise},
    {"space_lower", space_lower},
    {"laser_amplitude_modulation", laser_amplitude_modulation},
    {"laser_frequency_modulation", laser_frequency_modulation},
    {"susceptibility", susceptibility},
    {"force_calculation_left", force_calculation_left},
    {"corrected_optomechanical_phase_left", corrected_optomechanical_phase_left},
    {"corrected_optomechanical_phase_right", corrected_optomechanical_phase_right},
    {"space_modulation_lower", space_modulation_lower},
    {"squeezer", squeezer},
    {"laser_frequency_modulation_lower", laser_frequency_modulation_lower},
};
const int TABLE_SIZE = sizeof(TABLE) / sizeof(TABLE[0]);

}  // namespace

extern "C" {

// The identifier of a component function by its name in components.py, -1 if it is not here.
int ent_function(const char* name) {
    for (int k = 0; k < TABLE_SIZE; k++)
        if (std::strcmp(TABLE[k].name, name) == 0) return k;
    return -1;
}

// Evaluates one component function for `members` calls and `rows` rows of the sweep.
//
//   inputs[j]    values of input j: members * rows doubles (call by call) if swept[j], else members
//   direction[j] the derivative direction of input j (0 .. width - 1), or -1 for a plain input
//   slot[o]      where output o is stored (0 .. stored - 1), or -1 for an output nobody reads
//   values       stored * members * rows complex numbers: call c, row r of slot k at [(k * members + c) * rows + r]
//   slopes       stored * width * members * rows: direction i of slot k at [((k * width + i) * members + c) * rows + r]
//   flags[o]     bit 0: the output has a derivative, bit 1: its value differs along the sweep,
//                bit 2: its derivative differs along the sweep
//
// Returns 0, or -1 for an unknown function or too many directions.
int ent_eval(int function, int members, int rows, int width, const double* const* inputs, const int* swept,
             const int* direction, const int* slot, cplx* values, cplx* slopes, int* flags) {
    if (function < 0 || function >= TABLE_SIZE || width < 0 || width > MAXW) return -1;
    W = width;
    D p[7], out[4];
    for (int j = 0; j < 7; j++) {
        p[j].real = true;
        p[j].vs = swept[j] != 0;
        p[j].active = direction[j] >= 0;
        p[j].ts = false;
        for (int i = 0; i < MAXW; i++) p[j].t[i] = 0.0;
        if (p[j].active) p[j].t[direction[j]] = 1.0;
    }
    for (int o = 0; o < 4; o++) flags[o] = 0;
    for (int c = 0; c < members; c++) {
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < 7; j++) p[j].v = swept[j] ? inputs[j][(size_t)c * rows + r] : inputs[j][c];
            for (int o = 0; o < 4; o++) out[o] = D(0.0);
            TABLE[function].function(p, out);
            for (int o = 0; o < 4; o++) {
                if (slot[o] < 0) continue;
                size_t k = (size_t)slot[o];
                values[(k * members + c) * rows + r] = out[o].v;
                if (out[o].active)
                    for (int i = 0; i < width; i++) slopes[((k * width + i) * members + c) * rows + r] = out[o].t[i];
            }
        }
    }
    if (members > 0 && rows > 0)
        for (int o = 0; o < 4; o++) flags[o] = (out[o].active ? 1 : 0) | (out[o].vs ? 2 : 0) | (out[o].ts ? 4 : 0);
    return 0;
}

}  // extern "C"
