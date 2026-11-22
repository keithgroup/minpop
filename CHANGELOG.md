# Changelog

## 2.4.0 (2026-11-22)

Added
- Checkpoint files for GVB-PP. `minpop-gvbpp` writes the reference RHF/ROHF and the GVB-PP wavefunction (natural orbitals, pair occupations and overlaps, energies, core / pair / open counts) to a PySCF chkfile: `-chk FILE`, default `<xyz name>_gvbpp.chk`, `-no-chk` to skip.
- `minpop-gvbpp -read-chk FILE` (and `minpop-chk FILE`) reprojects a stored GVB-PP wavefunction with no SCF and no GVB-PP optimization. `-xyz` and `-npairs` are optional; `-minimal-basis`, `-no-keep` and the other projection flags apply. Works without the `gvbpp` package.
- `minpop-gvbpp -guess-chk FILE` starts the reference SCF from the file and, for a GVB-PP checkpoint with the same pair count, the GVB-PP optimization from its natural orbitals (new start `checkpoint`, `-gvb-guess checkpoint`). The stored start replaces the multi-start unless `-gvb-guess` or `-gvb-trials` is given.
- A GVB-PP checkpoint also serves as `-guess-chk` for the HF and KS drivers (its reference orbitals).
- `minpop-chk FILE -info` reports the GVB-PP energy, correlation energy, orbital space and pair occupations.
- `run_gvbpp_from_xyz(..., chk=, read_chk=, guess_chk=)` in the Python API.

Changed
- `minpop_gvbpp.py` imports `gvbpp` lazily and names the missing package only when a GVB-PP run is requested.

## 2.3.0 (2026-11-22)

Added
- `-guess-chk FILE` on the ROHF, UHF, RKS and UKS drivers: run the SCF from the wavefunction in a checkpoint (`minpop_chk.guess_density`). The file may differ in basis, method, functional, restricted/unrestricted form and frame; atoms, charge and multiplicity must match. Stored occupied orbitals are rotated (spherical and Cartesian shells), projected and orthonormalized as needed. Replaces `-guess`, `-guessmix`, `-guess-triplet` for that run.
- The report prints `Initial guess: checkpoint file ...`; `minpop-chk FILE -info` shows which file an SCF was started from.
- JSON schema 1.4: `source.guess_checkpoint`. Additive over 1.3.
- `run_*_from_xyz(..., guess_chk=)` in the Python API.

## 2.2.0 (2026-11-22)

Added
- Checkpoint files (`minpop/minpop_chk.py`). The ROHF, UHF, RKS and UKS drivers now obtain their wavefunction through this module. Each run writes the converged SCF to a PySCF chkfile (`-chk FILE`; default `<xyz name>_<method>.chk`; `-no-chk` to skip). `-read-chk FILE` skips the SCF and projects the stored wavefunction, so one SCF serves any number of minimal bases. `-xyz` is optional with `-read-chk`.
- `minpop-chk FILE ...`: reprojects a checkpoint with the driver that wrote it; `-info` prints what the file holds; `-method` selects the driver for a plain PySCF chkfile.
- A reprojection prints the stored SCF section verbatim, so its report and JSON record match a direct run with the same minimal basis apart from the command line and the `Checkpoint file:` line.
- Guards: SCF-defining flags (`-charge`, `-mult`, `-basis`, `-basis-light`, `-cartesian`, `-x2c`, `-x2c-atomic`, `-xc`, `-grid-level`, `-ecp`) and an `-xyz` given with `-read-chk` must agree with the file; a wavefunction of another method is refused with the right command named.
- JSON schema 1.3: `source.checkpoint = {path, mode}` (`written` or `read`), `null` without a checkpoint. Additive over 1.2.
- `run_*_from_xyz(..., chk=, read_chk=)` in the Python API.

Changed
- `minpop_rks.py`: the aufbau check uses O(n^3) matrix products instead of an O(n^5) einsum.
- `minpop_uhf.py`, `minpop_uks.py`: `-stable` follows the lowest mode of the orbital-rotation Hessian in both directions and keeps the lower solution; it now finds the spin-symmetry-breaking instability of a closed-shell start.

## 2.1.0 (2026-11-13)

Added
- JSON schema 1.2 (`minpop_json_csv.py`): the record now holds everything the report prints. New: `source.command_line`; `method.{label, basis_full, basis_heavy, basis_light, light_max_z, spherical, ecp, hamiltonian, xc, grid_level, minimal_basis, scf_solver}`; `scf.{s2, two_s_plus_one, n_doubly, n_singly}`; top-level `projection` (completeness lines), `symmetry` (σ/π/δ/φ matrices condensed to atoms and shells, total and spin), `gvb` (energies, trials, pair table), `warnings`. `method.name` also reports `rks`, `uks`, `gvbpp`. Atomic symbols cover the whole periodic table. Additive over 1.1.
- Long CSV carries the same new quantities (`projection_*`, `symmetry_*`, `gvb_*`, `s2`).
- `load_record` / `iter_jsonl` revive arrays at any depth.
- `minpop_gvbpp.py`: `-json`, `-json-xyz-dir`, `-csv-long`, `-csv-sparse-tol`; prints an occupation-weighted "Projection completeness (natural orbitals ...)" line.

Changed
- Repository moved to github.com/keithgroup/minpop.

## 2.0.0 (2026-10-27)

Added
- `-minimal-basis ano-rcc-mb` (H to Cm, all-electron) and `sto-3g-ano` (STO-3G / STO-3G* on Na–Ar for main-group atoms up to Ar, ANO-RCC-MB on every heavier atom; identical output to `sto-3g` for H–Ar molecules; alias of `sto-3g+ano-rcc-mb`).
- `minpop/basis_sets/anorcc_basis_pyscf.py`: ANO-RCC-MB/VDZP/VTZP/VQZP sliced from PySCF's ANO-RCC; verified against Basis Set Exchange.
- `-x2c` scalar-relativistic Hamiltonian in every driver; `-x2c-atomic` for atom-by-atom (DLU) decoupling.
- `-basis-light` / `-light-max-z`: per-element computational basis split with per-side ECP detection.
- `ano-rcc-mb-occ` / `sto-3g-ano-occ`: ANO-RCC-MB without the empty outermost p on non-p-block atoms beyond Ar (MINAO valence structure).
- `-minimal-basis LIGHT+HEAVY` (e.g. `sto-3g+ano-rcc-mb`, `sto-3g+minao`): per-element minimal basis with the same cutoff.
- `-symmetry` / `-symmetry-shells` (alias `-sigma-pi`): exact σ/π/δ/φ resolution of bond populations (`minpop/bond_symmetry.py`), printed as MBS matrices condensed to atoms and shells for the total and spin density.
- `minpop_gvbpp.py`: MinPop on GVB-PP natural orbitals (keithgroup/gvbpp) with occupation-weighted projection, representability rule for correlating NOs, valence-restricted pair search, multi-start with seed control, per-pair bond table.
- Valence-only minimal shells on ECP atoms are labelled with physical principal quantum numbers (MINAO on Rh with def2-ECP: 4S, 4P, 4D, 5S).
- "Projection completeness" diagnostic (captured electrons, leak, least-captured MO).
- `-label-style gaussian|physical`; Gaussian's labels remain the default for `sto-3g` and `sto-3g-ano`, physical labels are the default for `ano-rcc-mb`.
- Package layout with console commands (`minpop-rohf`, `minpop-uhf`, `minpop-rks`, `minpop-uks`, `minpop-gvbpp`, `minpop-export`), tests, examples.

Unchanged
- The STO-3G (Gaussian MBS) path is byte-identical to 1.x except for the added completeness line.

Known
- The legacy Gaussian shell-counter labels mark the 4d block of 5th-row atoms as "5D" where Gaussian prints "6D"; numbers and ordering are unaffected. Use `-label-style physical`.
