# External proton trap for ERKALE `hneoci`

A static, one-body external confining potential on the quantum proton, so that a
single electron + single proton (NEO-HF guess and `hneoci` FCI) can be solved
with the proton held in a tunable **cylindrical harmonic + quartic** trap. This
anchors an otherwise free 1e + 1p system into a contamination-free laboratory
for optimizing protonic and near-proton electronic Gaussian exponents.

Only the designated quantum proton feels the trap; the electron stays
Coulomb-bound to the trapped proton. The trap is added to the protonic core
Hamiltonian `H0p`, so it is seen by **both** the proton BO spectrum
("Quantum proton spectrum") and the CI.

## The operator

Coordinates are relative to the trap center `R0` (default: the quantum proton's
basis center). Atomic units throughout; `m` is the proton mass (`ProtonMass`,
default 1836.15267389).

```
V(r) = 1/2 m [ w_perp^2 (x^2 + y^2) + w_par^2 z^2 ]        (harmonic, axis = z)
     + lam_par z^4 + lam_perp (x^2+y^2)^2 + lam_cross z^2 (x^2+y^2)   (quartic)
```

The physical anharmonicity knob is the **dimensionless** `g` (`TrapG`), from
which the axis-resolved quartic couplings follow (each mode carries the same
relative anharmonicity):

```
lam_par  = g m^2 w_par^3
lam_perp = g m^2 w_perp^3
lam_cross = TrapLambdaCross   (independent input, default 0)
```

Isotropic quartic `lam r^4` corresponds to `(lam_par, lam_perp, lam_cross) =
(lam, lam, 2 lam)` because `r^4 = z^4 + (x^2+y^2)^2 + 2 z^2 (x^2+y^2)`.

Frequency unit conversion (when `TrapOmegaUnit cm-1`):
`w[Eh] = nu[cm^-1] * 4.5563352529e-6`.

## Matrix elements

`V` is a sum of Cartesian monomials `C_{abc} (x-X0)^a (y-Y0)^b (z-Z0)^c` of total
degree 2 or 4; its matrix in the protonic AO basis is a linear combination of
Cartesian moment integrals `M^{abc} = <mu| (x-X0)^a (y-Y0)^b (z-Z0)^c |nu>`,
formed by ERKALE's existing `BasisSet::moment(order, X0, Y0, Z0)` (order 4 =
15 moments), indexed by `getind(a,b,c)`:

```
V = 1/2 m w_perp^2 (M^{200} + M^{020})
  + 1/2 m w_par^2   M^{002}
  + lam_par         M^{004}
  + lam_perp        (M^{400} + 2 M^{220} + M^{040})
  + lam_cross       (M^{202} + M^{022})
```

`V` is real symmetric and enters `H0p = Tp + Vp + V` exactly like the protonic
kinetic term `Tp`; nothing about the two-particle (e-p) integrals, the SCF, or
the CI structure changes.

## Keywords (in the `hneoci` runfile)

| keyword           | meaning                                                        | default |
|-------------------|----------------------------------------------------------------|---------|
| `NEOTrap`         | master switch                                                  | `false` |
| `TrapOmegaUnit`   | unit of `TrapOmega*`: `cm-1` or `Eh`                           | `cm-1`  |
| `TrapOmegaPar`    | `w_par` (trap frequency along z)                               | `0.0`   |
| `TrapOmegaPerp`   | `w_perp` (perpendicular frequency; set `== TrapOmegaPar` for isotropic) | `0.0` |
| `TrapG`           | dimensionless anharmonicity `g` (sets `lam_par`, `lam_perp`)   | `0.0`   |
| `TrapLambdaCross` | `z^2 (x^2+y^2)` coefficient (a.u.); either sign is allowed      | `0.0`   |
| `TrapCenter`      | `x y z` in bohr; empty = quantum proton center                | (proton center) |

Per-state protonic moments about the trap center, multi-root energies and state
averaging are documented in [`README_MOMENTS.md`](README_MOMENTS.md).

`NEOTrap false` (or `NEOTrap true` with `TrapG 0` and `w = 0`) reproduces the
trap-off `hneoci` result bit-for-bit (`build_proton_trap` returns a zero
matrix when disabled).

**v1 restriction:** a single quantum proton. With more than one protonic
nucleus, set `TrapCenter` explicitly (otherwise the writer throws).

## Validation (`tests/trap_tests.sh`)

Analytic checks on the printed "Quantum proton spectrum" (the eigenvalues of the
one-body `H0p`, i.e. the trapped proton without the electron), so no external
reference code is needed for T0-T3:

- **T0 — regression.** `NEOTrap false` reproduces the baseline energies exactly;
  `V` is symmetric.
- **T1 — isotropic harmonic.** A single protonic `s` primitive with exponent
  `zeta = m w / 2` in an isotropic trap gives proton ground energy `(3/2) w`.
- **T2 — cylindrical harmonic.** `w_par != w_perp` with a converged basis:
  ground state `-> w_perp + 1/2 w_par`; ladder = doubly degenerate `pi` at
  `w_perp`, `sigma` at `w_par`, integer overtones.
- **T3 — quartic.** In a single matched Gaussian the proton energy *is* the
  expectation value, so the quartic shift is exact for any `g`, not just
  perturbatively. Isotropic `g` gives `Delta E = (11/4) g w` (pinning `M^{004}`,
  `M^{400}`, `M^{040}` and the factor 2 on `M^{220}`), and a pure cross term
  gives `Delta E = lam_cross / (2 m^2 w^2)` (pinning `M^{202} + M^{022}`).
  A single axis-resolved `g` also drives `lam_perp`, so the transverse motion
  cannot be frozen through the keyword set; these exact expectation values are a
  stronger check than the perturbative 1D shift `(3/4) g w_par` would be.

## Files touched

`src/contrib/hneoci.cpp` only: the keywords, the `build_proton_trap()` matrix
assembly, and `H0p = Tp + Vp + trap.V`. No changes to `neo.cpp`, the two-particle
integrals, the optimizer, or the electronic basis handling.
