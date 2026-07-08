# Per-state protonic moments and state averaging in `hneoci`

`erkale_hneoci` solves a single electron plus a single quantum proton by full CI.
This document covers the multi-root analysis layered on top of that: for the low
CI manifold it reports each root's energy, excitation gap and **protonic density
moments** ⟨z²⟩, ⟨x²+y²⟩, ⟨z⁴⟩, plus a **state-averaged** energy, moment set and
1-RDM. A machine-readable `.props.json` sidecar carries all of it, so a sweep
driver parses data rather than prose.

The confining trap these moments are usually measured against is documented
separately in [`README_TRAP.md`](README_TRAP.md).

## What "state averaging" means here

For a 1e + 1p system `hneoci`'s CISD is FCI, and **FCI roots and their densities
are invariant to the choice of reference orbitals**. One diagonalization of the
CI Hamiltonian in the fixed one-particle space yields every root and every
eigenvector, whether or not the underlying orbitals were state-averaged.

So there is no state-averaged MCSCF here, and there is nothing to gain from one.
"State averaging" is strictly a **post-diagonalization weighted average over the
CI roots** — an optimization target, not an orbital-optimization scheme. Were the
CI ever truncated rather than full, SA orbitals would begin to matter; that is
out of scope and flagged in the code.

## The protonic 1-RDM

The CI vector of root `I` has coefficients `c^I_{i,P}` over products of an
electronic and a protonic orbital. Tracing out the electron gives the protonic
one-particle density matrix in the protonic MO basis,

```
D^{I,p}_{PQ} = sum_i c^I_{i,P} c^I_{i,Q}
```

which the protonic MO coefficients back-transform to the AO basis,
`D^{I,AO} = Cp D^{I,p} Cp^T`. For a normalized single-proton root
`Tr D^{I,p} = 1`, hence `Tr[D^{I,AO} S_p] = 1` — reported per root as
`Tr[D S_p]`, and a sharp check on the tracing.

The moment operators are exactly the Cartesian moment matrices the trap is
assembled from, taken about the **same center** `R0` (`TrapCenter`, or the
quantum proton's basis center):

```
<z^2>_I     = Tr[ D^{I,AO} M^{002} ]
<x^2+y^2>_I = Tr[ D^{I,AO} (M^{200} + M^{020}) ]
<z^4>_I     = Tr[ D^{I,AO} M^{004} ]
```

No new integrals are computed: `build_proton_trap` now returns the moment
matrices it already formed, and they are formed even with the trap off whenever
the moments are requested. `MomentsFull` additionally reports
⟨x²⟩, ⟨y²⟩, ⟨x⁴⟩, ⟨y⁴⟩, ⟨x²y²⟩, ⟨x²z²⟩, ⟨y²z²⟩ — free once `D^{I,AO}` is in hand.

The state average is formed as `D_avg = sum_I w_I D^{I,AO}` and contracted once;
this is identical to averaging the per-root moments, since the contraction is
linear in `D`.

## Keywords

| keyword        | meaning                                                             | default |
|----------------|---------------------------------------------------------------------|---------|
| `NRoots`       | number of CI roots to solve, store and print                        | `1`     |
| `PrintMoments` | per-state protonic moments ⟨z²⟩, ⟨x²+y²⟩, ⟨z⁴⟩                      | `false` |
| `MomentsFull`  | also print the full order-2 / order-4 moment set                    | `false` |
| `StateAverage` | report the weighted `E_avg` and moment averages                     | `false` |
| `SAWeights`    | per-root weights, renormalized; shorter list zero-pads (empty = equal) | (equal) |
| `SADensityOut` | write the state-averaged protonic 1-RDM in the AO basis (implies `StateAverage`) | `false` |

**Regression guard.** With the defaults (`NRoots 1`, `PrintMoments false`,
`StateAverage false`) no multi-root block is printed, no moment integrals are
formed, and no sidecar is written — the numerical output is exactly that of the
pre-change `hneoci`. The only textual difference is that `settings.print()`
echoes the six new keywords, as it does for every registered keyword.

## Output

The per-root table and the average block go to stdout:

```
 root         E (Eh)             gap (Eh)         Tr[D S_p]        <z^2>         <x^2+y^2>        <z^4>
    0   0.013405730965   0.000000000000    1.00000000  2.72308511e-02 5.44617021e-02 2.22455775e-03
    1   0.023405731023   0.010000000058    1.00000000  2.72308511e-02 1.08923404e-01 2.22455775e-03
...
State average over 4 roots, SAWeights weights: 0.000000 0.333333 0.333333 0.333333
  E_avg =  0.023405731023
  <z^2>_avg = 4.53847518e-02  <x^2+y^2>_avg = 9.07695036e-02  <z^4>_avg = 5.19063475e-03
```

Alongside them, `<runfile>.props.json` — energies in Hartree, moments in bohr² /
bohr⁴, all numbers written at round-trip double precision (`%.17g`):

```json
{
  "trap": {"enabled": true, "omega_par_au": 0.01, "omega_perp_au": 0.01,
           "g": 0, "lambda_cross_au": 0, "center": [0, 0, 0]},
  "roots": [
    {"index": 0, "E": 0.0134057, "gap": 0, "trace": 1, "z2": 0.0272308, "x2py2": 0.0544617, "z4": 0.00222456}
  ],
  "state_average": {"weights": [0, 0.3333, 0.3333, 0.3333], "E": 0.0234057,
                    "z2": 0.0453848, "x2py2": 0.0907695, "z4": 0.00519063,
                    "density_file": "..."}
}
```

`MomentsFull` adds `x2`, `y2`, `x4`, `y4`, `x2y2`, `x2z2`, `y2z2` to each record
(note `x2py2` = ⟨x²+y²⟩ versus `x2y2` = ⟨x²y²⟩). `SADensityOut` writes
`<runfile>.sadens.dat`, the AO-basis `D_avg` as a plain ASCII matrix
(`numpy.loadtxt`-readable), and records its path in the sidecar.

## A caveat on degenerate roots

Within a degenerate manifold the individual CI eigenvectors are basis dependent
— LAPACK returns *some* orthonormal set spanning it. Per-state moments of
degenerate roots are therefore **not** physically meaningful on their own: for a
cylindrical trap ⟨x²⟩ = ⟨y²⟩ holds per state only for a *nondegenerate* state,
while inside a π doublet only quantities summed over the manifold are invariant.

This is why the validation below averages over the whole n=1 manifold rather
than testing an individual π root, and why the σ state is isolated with a
`TrapLambdaCross` splitting before its moments are compared.

## Validation (`tests/moment_tests.sh`, checks in `tests/moment_check.py`)

Every reference below is **exact to machine precision** — 38/38 assertions pass
with residuals at the 1e-15 level. Two constructions make that possible.

*Decoupling the electron.* The electron is given a single, very diffuse `s`
primitive (ζ = 10⁻⁶). Its density is then flat across the proton's extent, so
the proton feels the trap plus a constant. Crucially, that electronic density is
*spherical*, so the effective potential it exerts on the proton is radial: it
cannot couple protonic states of different parity. The protonic CI states are
therefore exactly the harmonic-oscillator states.

*A protonic basis that is exact, not merely converged.* A single `s` and a single
`p` primitive at ζ = m w / 2 represent the n = 0 and n = 1 isotropic-oscillator
states **exactly**. (An *anisotropic* trap cannot be treated this way: its
eigenfunctions need l = 0, 2, 4, … and isotropic Gaussians converge to them only
slowly — an ET s–f set still leaves ~6 % error in ⟨z²⟩ while the gaps are already
within 1 %. Moments converge much more slowly than energies.)

With `b = m w`:

- **V1 — ground-state moments.** ⟨z²⟩₀ = 1/(2b), ⟨x²+y²⟩₀ = 1/b, ⟨z⁴⟩₀ = 3/(4b²).
- **V2 — excited-state moments.** A negative `TrapLambdaCross` pushes σ_z below
  the π doublet. Parity keeps the trap block-diagonal in {s, pₓ, p_y, p_z}, so
  the splitting **reorders the states without mixing them**: root 1 is exactly
  the σ_z oscillator state, with ⟨z²⟩ = 3/(2b), ⟨z⁴⟩ = 15/(4b²), ⟨x²+y²⟩ = 1/b.
  The σ/π splitting itself equals |λ_cross|/(2b²), which independently pins the
  sign and scale of the `M202 + M022` assembly. Confirms the *excited-state* RDM,
  not just the ground state.
- **V3 — gaps.** The isotropic n = 1 manifold sits exactly `w` above the ground
  root. (The anisotropic ladder — π at `w_perp`, σ at `w_par` — is brief-1's T2
  in `tests/trap_tests.sh`.)
- **V4 — RDM trace.** `Tr[D^{I,AO} S_p] = 1` for every root. Catches any
  normalization or tracing error.
- **V5 — averaging.** Weights `(1,0,0,0)` reproduce the root-0 values exactly;
  empty `SAWeights` gives the arithmetic mean; the degeneracy-invariant
  equal-weight average over the n = 1 manifold gives ⟨z²⟩ = 5/(6b),
  ⟨x²+y²⟩ = 5/(3b), ⟨z⁴⟩ = 7/(4b²). `D_avg` is symmetric, and ⟨x²⟩ = ⟨y²⟩ both
  for the nondegenerate ground root and summed over the π manifold.
- **V6 — regression.** A default run prints no multi-root block and writes no
  sidecar; enabling the machinery leaves the `CI energy` bit-identical.

Run it with:

```bash
tests/moment_tests.sh                 # autodetects openmp/ or serial/ build
tests/moment_tests.sh /path/to/erkale_hneoci
```

## Files touched

`src/contrib/hneoci.cpp` only. `build_proton_trap` now returns a `proton_trap_t`
carrying the trap center and its moment matrices (previously it returned the bare
potential and discarded the moments); everything else is new code —
`proton_rdm`, `proton_moments`, `sa_weights`, `write_props`, and the reporting
block. The CI Hamiltonian build, the trap operator, the integrals and the orbital
handling are unchanged.

One pre-existing defect was fixed in passing: `TrapLambdaCross` was registered
with ERKALE's default `negative=false`, so `settings` rejected negative values.
λ_cross is an independent coefficient of either sign — only the total `V` need
stay confining — and negative values are needed to push σ_z below π. It is now
registered with `negative=true`.
