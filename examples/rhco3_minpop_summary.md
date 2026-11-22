# MinPop on Rh(CO)₃: choosing the computational and minimal bases

Executive summary, October 28, 2026. Barbaro Zulueta.

## Purpose

Decide how to run MinPop on transition-metal complexes: which computational basis on the metal and on the ligands, whether to treat relativity with X2C or an ECP, and which minimal basis to project onto. Rh(CO)₃ (neutral, doublet, UPBE, C₃v test geometry) served as the test case. Eleven runs cover the combinations; eight are valid and compared below. Every run uses the same geometry and functional, so all differences are partitioning or basis effects.

## Settings

| Label | Rh computational basis | Ligand basis | Relativity | Minimal basis (Rh / C,O) | Wall time |
|---|---|---|---|---|---|
| 4 | ANO-RCC-VTZP | def2-TZVPP | full X2C | ANO-RCC-MB / ANO-RCC-MB | 2 h |
| 5 | ANO-RCC-VTZP | def2-TZVPP | atomic X2C | ANO-RCC-MB / ANO-RCC-MB | 20–30 min |
| 6 | ANO-RCC-VTZP | def2-TZVPP | atomic X2C | ANO-RCC-MB-OCC / ANO-RCC-MB | 20–30 min |
| 7 | ANO-RCC-VTZP | def2-TZVPP | atomic X2C | ANO-RCC-MB-OCC / STO-3G | 20–30 min |
| 8 | ANO-RCC-VTZP | def2-TZVPP | atomic X2C | ANO-RCC-MB-OCC / MINAO | 20–30 min |
| 9 | x2c-TZVPPall | def2-TZVPP | atomic X2C | ANO-RCC-MB / ANO-RCC-MB | 8–15 min |
| 10 | ANO-RCC-VTZP | 6-31G* | atomic X2C | ANO-RCC-MB / ANO-RCC-MB | 10–15 min |
| 11 | x2c-TZVPPall | 6-31G* | atomic X2C | ANO-RCC-MB / ANO-RCC-MB | 8–10 min |
| 1 | def2-TZVPP + def2-ECP(28) | def2-TZVPP | ECP | MINAO / STO-3G | 6–8 min |
| 2 | def2-TZVPP + def2-ECP(28) | def2-TZVPP | ECP | MINAO / MINAO | 6–8 min |

Two runs were discarded: an all-electron minimal basis (ANO-RCC-MB) on Rh with the ECP calculation, which books 1.3 e of valence density to core functions, and a run where `-x2c-atomic` was given without `-x2c` in an earlier code version (now fixed; the flag implies X2C).

ANO-RCC-MB-OCC is ANO-RCC-MB without the empty outermost p shell on atoms beyond Ar (Rh 5s3p2d instead of 5s4p2d). It has the valence shell structure of PySCF's MINAO for every element MINAO defines, extended by the same rule to the rest.

## Results

| Case | q(Rh) | spin Rh / C2 | q(C2) / q(O) | Rh–C σ (C2, C4) | Rh–C π | Rh 4d | Rh 5p | leak |
|---|---|---|---|---|---|---|---|---|
| 4 | +0.04 | 0.72 / 0.12 | +0.37 / −0.37 | +0.17, +0.19 | 0.13 | 7.94 | 0.75 | 0.39 % |
| 5 | +0.04 | 0.72 / 0.12 | +0.37 / −0.37 | +0.17, +0.19 | 0.13 | 7.94 | 0.75 | 0.39 % |
| 9 | +0.05 | 0.71 / 0.12 | +0.37 / −0.38 | +0.17, +0.19 | 0.13 | 7.94 | 0.75 | 0.39 % |
| 10 | +0.03 | 0.72 / 0.12 | +0.35 / −0.36 | +0.17, +0.19 | 0.13 | 7.95 | 0.75 | 0.42 % |
| 11 | +0.05 | 0.72 / 0.12 | +0.35 / −0.36 | +0.17, +0.19 | 0.13 | 7.94 | 0.75 | 0.41 % |
| 6 | +0.65 | 0.56 / 0.20 | +0.21 / −0.39 | 0.00, −0.06 | 0.12 | 8.08 | — | 0.65 % |
| 8 | +0.73 | 0.56 / 0.20 | +0.19 / −0.40 | −0.02, −0.08 | 0.12 | 8.06 | — | 0.61 % |
| 7 | +0.42 | 0.57 / 0.19 | +0.13 / −0.23 | +0.04, −0.03 | 0.10 | 8.09 | — | 0.95 % |
| 2 | +0.93 | 0.56 / 0.19 | +0.13 / −0.41 | −0.07, −0.13 | 0.11 | 8.02 | — | 0.97 % |
| 1 | +0.68 | 0.57 / 0.19 | +0.05 / −0.23 | 0.00, −0.07 | 0.10 | 8.12 | — | 1.56 % |

Leak is the share of occupied density the minimal basis could not represent before renormalization.

## Findings, in order of size

1. **The empty Rh 5p in the minimal basis is the largest effect: 0.6 e.** With it (cases 4, 5, 9, 10, 11) Rh reads +0.04, the 5p holds 0.75 e of carbonyl σ density, and the Rh–C σ terms are +0.17 to +0.19. Without it (cases 6, 7, 8) Rh reads +0.4 to +0.7, the σ terms are near zero or negative, and the metal picture matches the ECP/MINAO runs. This is a convention, not an error, but it decides what "the charge on Rh" means.

2. **The ligand minimal basis is the second effect: 0.15 e on O, 0.25 e on Rh.** STO-3G (cases 7, 1) versus free-atom functions, ANO-RCC-MB or MINAO (cases 6, 8, 2). MINAO and ANO-RCC-MB on the ligands are interchangeable (case 6 vs 8: 0.07 on Rh, 0.01 on O). The same pattern was measured on H₂O, CO and LiF: free-atom minimal functions reach the neighbour and read polar bonds more ionic. Both the leak and the partition move together.

3. **All-electron X2C versus ECP on the metal, at matched shell structure: 0.25 e on Rh, 0.05 on σ.** Case 7 vs 1 and case 8 vs 2. Spins, 4d populations, π back-donation and the Jahn-Teller pattern agree to 0.003. This is the irreducible difference between nodal ANO valence functions and MINAO's pseudo-valence functions.

4. **The computational basis does not matter once it is triple-zeta on the metal and polarized on the ligands.** Cases 4, 9, 10, 11 agree to 0.02 e on every population and to 0.003 on every bond term while the energies spread over 0.26 Eh. x2c-TZVPPall reproduces ANO-RCC-VTZP; 6-31G* reproduces def2-TZVPP to 0.017 on O.

5. **Atomic X2C equals full X2C.** Case 5 reproduces case 4 to 2 × 10⁻⁶ in every population at a fifth of the cost. Relativity cannot be restricted to one element; the atomic decoupling is the economical equivalent.

6. **What is invariant across all eight valid runs:** Rh 4d ≈ 8.0, spin 0.56 to 0.57 on Rh and 0.19 on the unique carbon (0.72 / 0.12 with the 5p convention), π back-donation 0.10 to 0.13 per carbonyl, σ donation netting to zero or slightly negative against the filled 4s4p semicore, and a SOMO of Rh 4d character with a σ* tail on one carbonyl. That is the physics; the rest of the table is where the partition line falls.

7. **Projection completeness.** 0.4 to 1.0 % all-electron, 1.6 % for the STO-3G-ligand ECP run. The missing density sits on the C≡O π and σ orbitals and the SOMO (least-captured orbital 90 to 96 %), never on cores. Not a concern for the conclusions, but it should be quoted.

## Recommendations

**Production procedure for metal complexes with main-group ligands**

```
minpop-uks -xyz geom.xyz -charge Q -mult M -xc pbe -stable -symmetry-shells \
    -basis x2c-tzvppall -basis-light "6-31G*" -x2c-atomic -minimal-basis sto-3g-ano-occ
```

- Computational basis: x2c-TZVPPall on atoms beyond Ar, 6-31G* on H–Ar (def2-TZVPP on the ligands when the molecule is small). Same populations as ANO-RCC-VTZP/def2-TZVPP at a tenth of the cost.
- Relativity: `-x2c-atomic`. Same result as full X2C, linear cost.
- Minimal basis: `sto-3g-ano-occ`, which is STO-3G (STO-3G* on Na–Ar) on the ligands and ANO-RCC-MB-OCC on the metal. The ligands keep the convention of the existing STO-3G/Gaussian MinPop literature; the metal gets a shell structure that matches MINAO, so results are comparable with ECP work.
- Report alongside the charges: the symmetry-resolved Rh–L σ/π/δ matrices and the completeness line.

**When an ECP is unavoidable** (very large systems, or basis sets only available with ECPs): def2-TZVPP with def2-ECP and `-minimal-basis sto-3g+minao`. Never project an ECP calculation onto an all-electron minimal basis; the driver warns. Expect the metal 0.2 to 0.3 e more positive than the all-electron route and say so.

**Conventions to fix and keep fixed within a paper:** (a) with or without the empty valence p on the metal (recommend without, i.e. `-occ`); (b) STO-3G or free-atom functions on the ligands (recommend STO-3G for continuity, ANO-RCC-MB if every atom should be treated by one rule); (c) all-electron or ECP on the metal. Each of these moves numbers by 0.2 to 0.6 e; the computational basis does not.

**Do not use** MINAO as a general minimal basis (valence-only beyond Kr, no entries for K, Rb, Sr, Cs, Ba and the f block, drops the empty p on Li, Be, Na, Mg, no 3d on Na–Ar).

**Use `ano-rcc-mb` on a metal only when the 5p convention is intended.** Roos's minimal basis includes the empty valence p on every metal (5p on Rh), as Gaussian's STO-3G does for 3d and 4d metals. That function is diffuse and empty in the free atom, so in the complex it has nothing of its own to hold and fills with whatever ligand density reaches it: on Rh(CO)₃ it takes 0.75 e, all of it carbonyl σ lone-pair density. Mulliken then counts that density as belonging to Rh. The consequences are visible in the table: the Rh charge drops from +0.65 to +0.04, the Rh–C σ bond populations turn positive (+0.17 to +0.19) because the σ donation is now booked as shared density between the carbonyl and the 5p rather than as repulsion against the filled 4s4p semicore, and the Rh spin rises to 0.72 because the SOMO gains 5p character on paper. None of this reflects a change in the wavefunction; cases 4 and 6 are the same SCF. So `ano-rcc-mb` on a metal is a legitimate convention (it is the one Gaussian's MBS uses for 3d metals, and it is the right choice when one wants σ donation into the metal valence shell to appear as a bond term), but its metal charges and σ terms are not comparable with MINAO, with `ano-rcc-mb-occ`, or with the ECP literature. If the charge on the metal is the quantity being reported, or if any comparison with ECP/MINAO work is intended, use the `-occ` variant and say so.

## Caveats on this test case

- The geometry is a C₃v guess. The neutral doublet is a ²E state there and Jahn-Teller unstable; every run breaks the symmetry electronically (C2 ≠ C4, C6; spin 0.19 vs 0.06). Optimize the geometry at UPBE before any numbers are published; the three carbonyls will become inequivalent by geometry rather than by SCF symmetry breaking.
