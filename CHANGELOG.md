# Changelog

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
