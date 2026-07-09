# `neo_dump.h5` — NEO-SCF export format

A self-describing HDF5 dump of a **converged ERKALE NEO-SCF** (nuclear-electronic
orbital SCF), produced by `erkale_neo` when the `NEODump` keyword is set. It
contains everything a post-SCF correlation code (e.g. a spin-orbital NEO-CCSD
driver) needs: AO basis dimensions, MO coefficients, orbital energies, core and
Fock matrices, and the **bare AO two-particle integrals** (electron-electron,
proton-proton, electron-proton).

All quantities are in **Hartree atomic units**. Everything is in the **AO basis**
except `C` / `eps` (MO quantities). The downstream code performs the AO→MO
transformation and the spin-orbital construction itself; ERKALE's side is a pure
export.

Supported: **restricted (RHF) and unrestricted (UHF) electrons**, one or more
**quantum protons** (point nuclei). See *Limitations* for what is out of scope.

---

## 1. Conventions (read this first — the whole contract lives here)

### 1.1 Integral convention

All two-particle integrals use the **chemist (charge-cloud) convention** `(pq|rs)`,
the **bare (positive) Coulomb** integral over the real AO basis functions:

```
(pq | rs)  =  ∫∫  φ_p(r1) φ_q(r1)  (1 / |r1 - r2|)  φ_r(r2) φ_s(r2)  dr1 dr2
```

- Operator is the bare `1/r12`. **No charge factors and no sign are applied** — every
  stored integral is a positive Coulomb integral. Charge signs (electron `-1`,
  proton `+1`, or any other quantum species such as `µ⁻`) are the consumer's job.
- `p,q` share coordinate `r1`; `r,s` share `r2`.
- Basis functions are ERKALE's real solid-harmonic Gaussians, in the same AO order
  used by every matrix/tensor in this file. Treat the AO index as an opaque,
  self-consistent label.

Recorded in `/meta` as `integral_convention = "chemist (pq|rs)"`.

### 1.2 Storage / index order (CRITICAL)

Every array is written **C / row-major**, so an `h5py`/NumPy reader gets the natural
shape with **no transpose**: a dataset declared `(d0,d1,…)` reads back as `A` with
`A[i,j,…] = M(i,j,…)`. (ERKALE's internal `.chk` writer is column-major; this dump
deliberately is not — do not assume the `.chk` convention here.)

### 1.3 Two-particle integral representation

Selected by `/meta:integral_representation`:

**`"btensor"` (default).** The per-species integrals are stored *factorized*, exactly
as the SCF's density-fitting / Cholesky engine used them:

| Dataset       | Shape (C-order)        | Reconstruction |
|---------------|------------------------|----------------|
| `electron/B`  | `(naux_e, nbf_e, nbf_e)` | `(μν\|λσ) = Σ_P B[P,μ,ν]·B[P,λ,σ]` |
| `proton/B`    | `(naux_p, nbf_p, nbf_p)` | `(ab\|cd)  = Σ_P B[P,a,b]·B[P,c,d]` |

For density fitting `B = (μν|P)·V^{-1/2}` (metric `V=(P|Q)` of the shared aux basis);
for Cholesky `B = L` are the Cholesky vectors. The fit dimension `naux` (=
`/meta:naux_e`, `naux_p`) is the number of aux/Cholesky vectors and may differ
between species.

**`"dense"`.** The per-species integrals are stored as full rank-4 tensors:

| Dataset    | Shape (C-order)                 | Element |
|------------|---------------------------------|---------|
| `eri_ee`   | `(nbf_e,nbf_e,nbf_e,nbf_e)`     | `(μν\|λσ)` |
| `eri_pp`   | `(nbf_p,nbf_p,nbf_p,nbf_p)`     | `(ab\|cd)` |

These are reconstructed from the same engine factors (`= Σ_P B·B`), i.e. they carry
the SCF's fitting/Cholesky approximation, not a separate exact recomputation.

**Electron-proton — always dense, in both representations:**

| Dataset  | Shape (C-order)              | Element |
|----------|------------------------------|---------|
| `eri_ep` | `(nbf_e,nbf_e,nbf_p,nbf_p)`  | `(μν\|ab)`, electrons on `r1`, protons on `r2`, **bare positive** |

(In DF mode the SCF shares one aux basis, so `(μν|ab)=Σ_P B_e[P,μν]B_p[P,ab]`; in CD
mode the species have independent Cholesky spaces and the SCF computes e-p
*exactly* — so `eri_ep` is dumped dense to stay engine-consistent either way.)

### 1.4 MO ordering and occupations

- `C` holds the **SCF orbitals verbatim**, reordered so that the **occupied ones
  come first**. They are orthonormal (`Cᵀ S C = I`) but are **not** canonical
  with respect to any particular Fock matrix — see §2.3.
- `nmo ≤ nbf`: linearly dependent AO directions are removed by canonical
  orthogonalization, so `C` is `(nbf, nmo)` with no padding.
- Occupation is defined by **column order**: the first `nocc` columns are
  occupied, the rest are virtual. The `occ` vector stores the same information
  explicitly (and lets a reader assert it); the writer rejects fractional or
  non-aufbau occupations, which the column convention cannot express.

| Species block        | occ per occupied MO | `nocc` | density `D` |
|----------------------|---------------------|--------|-------------|
| electrons, RHF       | `2.0`               | `n_electrons/2` | `2·C[:,:nocc]C[:,:nocc]ᵀ` |
| electrons α / β, UHF | `1.0`               | `n_α` / `n_β`    | `C[:,:nocc]C[:,:nocc]ᵀ` |
| protons (high-spin)  | `1.0`               | `n_quantum_protons` | `C[:,:nocc]C[:,:nocc]ᵀ` |

`proton_spin_treatment = "high-spin"`: protons occupy a single spin block (one per
spatial orbital, full p-p exchange) — a fully spin-polarized determinant of identical
fermions. For one proton the p-p Coulomb+exchange self-interaction cancels exactly.

### 1.5 Sign of the electron-proton interaction

`eri_ep` is the **bare, positive** `(μν|ab)`. The physical interaction is attractive
(opposite charges); the dump does **not** apply that sign. The CC Hamiltonian
multiplies by `q_e·q_p = -1`:

```
E_ep  =  - Σ_{μν∈e}  Σ_{ab∈p}   eri_ep[μ,ν,a,b] · D^e[μ,ν] · D^p[a,b]
```

No Fock matrix is stored, so this file contains **no** sign-carrying mean-field
operator: every two-particle quantity in it is sign-free (§2.3).

---

## 2. File layout

### 2.1 `/meta` (root-group attributes)

| Attribute                  | Type   | Meaning |
|----------------------------|--------|---------|
| `units`                    | string | `"Hartree atomic units"` |
| `integral_convention`      | string | `"chemist (pq|rs)"` (§1.1) |
| `integral_representation`  | string | `"btensor"` or `"dense"` (§1.3) |
| `ao_or_mo`                 | string | `"AO"` |
| `storage_order`            | string | `"C (row-major)"` |
| `n_electrons`              | int    | total electrons |
| `restricted_electrons`     | bool   | `true` = RHF (one electron set); `false` = UHF (α/β sets) |
| `n_quantum_protons`        | int    | number of quantum protons |
| `proton_spin_treatment`    | string | `"high-spin"` |
| `proton_mass`              | double | proton mass (atomic units) scaling the nuclear kinetic operator |
| `naux_e`                   | int    | electron fit dimension (only in `btensor` mode) |
| `naux_p`                   | int    | proton fit dimension (only in `btensor` mode) |
| `e_scf`                    | double | converged NEO-SCF total energy printed by ERKALE |
| `e_classical`              | double | Coulomb repulsion among the *classical* (non-quantum) nuclei only |
| `erkale_version`           | string | ERKALE version / git description |

`e_classical` excludes quantum protons; their coupling to classical nuclei is a
one-particle term in `proton/hcore`, and to electrons it is `eri_ep`. No other
constants.

### 2.2 `/electron` and `/proton` (groups)

Common datasets:

| Dataset   | Shape         | Notes |
|-----------|---------------|-------|
| `nbf`     | scalar int    | AO basis functions |
| `overlap` | `(nbf,nbf)`   | AO overlap `S` — the metric this species' AO basis lives in |
| `B`       | `(naux,nbf,nbf)` | engine factor — `btensor` mode only (§1.3) |

Electron group, **RHF** (`restricted_electrons = true`):

| Dataset | Shape        | Notes |
|---------|--------------|-------|
| `nmo`,`nocc` | scalar int | `nocc = n_electrons/2` |
| `C`     | `(nbf,nmo)`  | SCF AO→MO, occupied columns first (§1.4, §2.3) |
| `occ`   | `(nmo,)`     | SCF occupations (`2.0` × `nocc`, then zeros) |
| `hcore` | `(nbf,nbf)`  | one-particle core (§2.3) |

Electron group, **UHF** (`restricted_electrons = false`): the same, suffixed `_a`/`_b`
— `nmo_a/nmo_b`, `nocc_a/nocc_b`, `C_a/C_b`, `occ_a/occ_b`, `hcore` (spin-independent,
single copy).

Proton group: `nmo`, `nocc` (`= n_quantum_protons`), `C`, `occ`, `hcore`.

### 2.3 What `hcore` contains, and why there is no `fock`

- **Electron `hcore`** `= T_e + V_e(classical)` — kinetic + attraction to the
  *classical* nuclei only. The e-p attraction is **not** here (two-particle, via
  `eri_ep`).
- **Proton `hcore`** `= T_p/m_p + V_p(classical)` — mass-scaled kinetic + repulsion
  from classical nuclei (`+` sign for the positive test charge), `m_p = proton_mass`.

**No Fock matrix and no orbital energies are dumped.** The right proton reference
operator is use-case dependent, and baking one in silently corrupts the density:

- A correlation model *without* a proton–proton fluctuation potential wants the
  **self-interaction-free** `f_p = hcore_p − J_pe[D^e]`.
- A model *with* one wants the **J/K-dressed** SCF Fock
  `f_p = hcore_p − J_pe[D^e] + J_pp[D^p] − K_pp[D^p]`.

These two operators share an occupied subspace **only for a single quantum proton**
(where `J_pp + K_pp` annihilates the occupied orbital by exact self-interaction
cancellation). For two or more protons they do not, so canonicalizing the SI-free
operator — as this writer used to do — yields an occupied subspace that is *not*
the SCF one, and hence a reconstructed density and reference energy that are wrong
(observed: 5.5 mHa for CH₄ with two quantum protons).

The dump therefore exports the **SCF orbitals themselves**, which define the SCF
density by construction, and leaves the choice of reference operator to the
consumer. The consumer:

1. builds `D^e`, `D^p` from the first `nocc` columns of `C` (§1.4);
2. builds whichever `f` it wants from `hcore`, `B` and `eri_ep`;
3. **semicanonicalizes** — diagonalizes `f` separately within the occupied and the
   virtual block — and takes `eps = diag(Cᵀ f C)`. This leaves both subspaces, and
   hence the density, untouched.

Step 3 is mandatory: the dumped orbitals are *not* eigenvectors of the SI-free
proton operator. `erkale_neo` prints `max|f_ia|`, the surviving occupied–virtual
coupling, at write time — it is at the SCF convergence level for one proton and
`O(10⁻²)` for two.

---

## 3. Energy reconstruction (self-consistency check)

```
E = Σ_species [ Tr(D h) + ½ Tr(D J[D]) − c_x Tr(D K[D]) ]
    − Σ_{μν∈e, ab∈p} eri_ep[μ,ν,a,b] · D^e[μν] · D^p[ab]
    + e_classical
```

with `J[D]_pq = Σ_rs (pq|rs) D_rs`, `K[D]_pq = Σ_rs (pr|qs) D_rs`, densities from
§1.4, and `c_x = ¼` for RHF electrons (total density, occ 2), `c_x = ½` for each UHF
spin and for the high-spin protons. The `(pq|rs)` come from `eri_xx` (`dense`) or are
reconstructed from `B` (`btensor`).

`erkale_neo` with `NEODumpVerify true` reconstructs `E` from the on-disk tensors and
asserts agreement with `e_scf`. `tests/neo_dump_check.py <dump.h5>` does the same from
outside ERKALE and is the reference implementation of the consumer contract: it builds
the densities from the dumped orbitals, rebuilds both reference Fock matrices,
semicanonicalizes them, and checks all of it. Run it against any new dump.

Because the dumped integrals match the SCF's own
engine (B-tensors / engine-reconstructed dense, and an engine-consistent `eri_ep`),
the residual is at machine precision for Cholesky (`≈1e-14`, any proton count) and
for RI with a single quantum proton.

**RI with ≥2 quantum protons** shows a residual of `≈1e-8`. This is a conditioning
artifact, not an inconsistency: `def2-universal-jkfit` is an *electronic* auxiliary
basis and fits the tight protonic pair densities very poorly. For CH₄/PB4-F1 the
proton fit factor `proton/B` has condition number `2.6e12` (vs `3.1e4` for Cholesky),
so the two algebraically-equivalent ways of contracting `J_pp` — via `B·Bᵀ` here, via
the metric solve in the SCF — differ at that level. With one proton the p-p Coulomb
and exchange cancel exactly, so the error is invisible; with two it is not.

Note also that the RI p-p energy itself is inaccurate, not merely ill-conditioned:
for CH₄ with two quantum protons, `J_pp` is `5.460` under RI against `5.750` under
Cholesky, a `0.29 Eh` fitting error that shifts the total energy by `1.1 mEh`. **Use
`JKMethod Cholesky` for NEO dumps with more than one quantum proton.** A future
superbasis Cholesky (one decomposition over the union of the electronic and protonic
bases, segmented into `B_e`/`B_p` sharing a common vector index) would remove both
problems and make `eri_ep` exact by construction rather than by aux-basis coincidence.

---

## 4. Limitations

- **Point protons only.** `FiniteProton` (range-separated / Gaussian nucleus) is not
  exported — the e-p operator would be screened, not bare `1/r12`. The writer throws.
- **Dense `eri_ep` / dense mode.** No permutational-symmetry packing — intended for
  small systems (one/two quantum protons, modest electronic basis).
- **RI with ≥2 quantum protons is inaccurate** (§3). Prefer `JKMethod Cholesky`.

---

## 5. Schema changelog

### v2 — core Hamiltonian + SCF orbitals (breaking)

Motivation: the writer used to store `C` as the canonical orbitals of the
self-interaction-free proton Fock. That operator does not share the SCF occupied
subspace once there is more than one quantum proton, so the reconstructed proton
density — and any correlation treatment built on it — was wrong. Measured on CH₄
with two quantum protons: `NEODumpVerify` off by `5.5e-3 Eh`. Single-proton dumps
were unaffected (exact self-interaction cancellation makes the two operators share
the occupied orbital), which is why it went unnoticed.

**Removed** (consumers must stop reading these):

| Dataset | Old shape | Replacement |
|---------|-----------|-------------|
| `electron/fock`, `electron/fock_a`, `electron/fock_b` | `(nbf,nbf)` | rebuild from `hcore` + `B` + `eri_ep` |
| `proton/fock`   | `(nbf,nbf)` | rebuild from `proton/hcore` + `eri_ep` (+ `proton/B` for a dressed Fock) |
| `electron/eps`, `electron/eps_a`, `electron/eps_b`   | `(nmo,)` | `diag(Cᵀ f C)` after semicanonicalization |
| `proton/eps`    | `(nmo,)` | `diag(Cᵀ f C)` after semicanonicalization |

**Added:**

| Dataset | Shape | Meaning |
|---------|-------|---------|
| `electron/occ` (or `occ_a`/`occ_b`) | `(nmo,)` | SCF occupations, `2.0`/`1.0` × `nocc` then zeros |
| `proton/occ` | `(nmo,)` | SCF occupations, `1.0` × `nocc` then zeros |

**Changed semantics** (same name, same shape):

| Dataset | Was | Is |
|---------|-----|-----|
| `electron/C`, `proton/C` | canonical eigenvectors of the dumped `fock` | the SCF orbitals, occupied columns first; orthonormal but **not** canonical |

Unchanged: `nbf`, `nmo`, `nocc`, `hcore`, `overlap`, `B`, `eri_ep`, `eri_ee`,
`eri_pp`, and every `/meta` attribute.

**Consumer migration.** A reader that previously did *load `F`, `eps` → assert
canonical → MO-transform* must now: build `D` from the first `nocc` columns of `C`;
build `f` itself (`f_e = h + J − ½K − V_pe`, `f_p = h − V_ep`, plus `J_pp − K_pp` if
the model carries a proton–proton fluctuation potential); **semicanonicalize** by
diagonalizing `f` within the occupied and virtual blocks separately, rotating `C`
accordingly; and take `eps = diag(Cᵀ f C)`. Asserting `‖f C − S C diag(eps)‖ ≈ 0` on
the *dumped* `C` will now fail by design — assert orthonormality instead, and assert
canonicality only after the semicanonical rotation.
