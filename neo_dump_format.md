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

- MOs in `C`/`eps` are **energy-ordered** (aufbau): occupied first, then virtual.
- `nmo ≤ nbf`: linearly dependent AO directions are removed by canonical
  orthogonalization, so `C` is `(nbf, nmo)` with no padding.
- Occupation vectors are **not stored** — they follow from `nocc` and the species rule:

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

The `fock` matrices in this file *do* already include the mean-field e-p attraction
with its `-` sign (they are the converged SCF Fock matrices — §2.3); only the raw
`eri_ep` tensor is sign-free.

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

| Dataset | Shape         | Notes |
|---------|---------------|-------|
| `nbf`   | scalar int    | AO basis functions |
| `B`     | `(naux,nbf,nbf)` | engine factor — `btensor` mode only (§1.3) |

Electron group, **RHF** (`restricted_electrons = true`):

| Dataset | Shape        | Notes |
|---------|--------------|-------|
| `nmo`,`nocc` | scalar int | `nocc = n_electrons/2` |
| `C`     | `(nbf,nmo)`  | AO→MO, energy-ordered |
| `eps`   | `(nmo,)`     | MO energies |
| `hcore` | `(nbf,nbf)`  | one-particle core (§2.3) |
| `fock`  | `(nbf,nbf)`  | converged Fock (§2.3) |

Electron group, **UHF** (`restricted_electrons = false`): the same, suffixed `_a`/`_b`
— `nmo_a/nmo_b`, `nocc_a/nocc_b`, `C_a/C_b`, `eps_a/eps_b`, `hcore` (spin-independent,
single copy), `fock_a/fock_b`.

Proton group: `nmo`, `nocc` (`= n_quantum_protons`), `C`, `eps`, `hcore`, `fock`.

### 2.3 What `hcore` and `fock` contain

- **Electron `hcore`** `= T_e + V_e(classical)` — kinetic + attraction to the
  *classical* nuclei only. The e-p attraction is **not** here (two-particle, via
  `eri_ep`).
- **Proton `hcore`** `= T_p/m_p + V_p(classical)` — mass-scaled kinetic + repulsion
  from classical nuclei (`+` sign for the positive test charge), `m_p = proton_mass`.
- **`fock`** is the converged self-consistent AO Fock with all mean-field two-particle
  terms at their physical signs (including the `-` e-p attraction). Its generalized
  eigenvalues with the AO overlap equal `eps`. Per species:
  `fock_e = hcore_e + J_ee[D^e] − ½K_ee[D^e] − J_ep[D^p]` (RHF; UHF uses
  `J_ee[D^e] − K_ee[D^{σ}]` per spin), `fock_p = hcore_p + J_pp[D^p] − K_pp[D^p]
  − J_pe[D^e]`, where `J_ep`/`J_pe` denote the (positive) Coulomb fields and the `−`
  is the attraction.

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
asserts agreement with `e_scf`. Because the dumped integrals match the SCF's own
engine (B-tensors / engine-reconstructed dense, and an engine-consistent `eri_ep`),
the residual is at the SCF convergence level (≈ `1e-9`) for both RI and Cholesky.

---

## 4. Limitations

- **Point protons only.** `FiniteProton` (range-separated / Gaussian nucleus) is not
  exported — the e-p operator would be screened, not bare `1/r12`. The writer throws.
- **Dense `eri_ep` / dense mode.** No permutational-symmetry packing — intended for
  small systems (one/two quantum protons, modest electronic basis).
