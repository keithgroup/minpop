#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026, Barbaro Zulueta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
MinPop population analysis on a GVB-PP wavefunction (keithgroup/gvbpp).

GVB-PP returns natural orbitals (NOs) with fractional occupations: core 2,
each pair a bonding NO with n_b and a correlating NO with n_c (n_b + n_c = 2),
open shells 1. The 1-RDM is diagonal in that basis, so MinPop generalizes in
one step: project the NOs onto the minimal basis, orthonormalize them there
(symmetric, as for HF), and build P = sum_i n_i |i><i|.

Representability. A minimal basis spans a bond's correlating orbital (the
sigma* of O-H lives in O 2p and H 1s) but not a lone pair's (that is a
3s/3d-like orbital). For H2O the nine NOs would not fit in seven functions.
So the NOs are projected in order of decreasing occupation and each one is
kept only if the part of it that is linearly independent of the NOs already
kept, measured in the minimal basis, exceeds the threshold -no-keep (default
0.5). A correlating NO that fails is dropped and its occupation n_c is
folded back into its bonding partner, so the electron count is exact and the
pair is treated at the level the minimal basis can represent: uncorrelated.
Core, bonding and open-shell NOs are always kept (a failure there means the
minimal basis is wrong for the system and is reported as an error).

Spin density comes from the open shells only; perfect pairing couples every
pair to a singlet.

The report is the regular MinPop report (same tables, order and formats),
preceded by a GVB-PP pair summary that lists, per pair, n_b, n_c, the GVB
overlap S_ab, whether its correlating NO was projected, and the pair's own
contribution to the MBS bond population between the two atoms it connects,
with that contribution's symmetry class.

Usage
-----
    python minpop_gvbpp.py -xyz pt2.xyz -mult 3 -npairs 2 \\
        -basis ano-rcc-vdzp -minimal-basis ano-rcc-mb -x2c -symmetry

Requires minpop_rohf.py, bond_symmetry.py and (for ANO-RCC) anorcc_basis_pyscf.py
next to this script, and the gvbpp package (pip install -e gvbpp).
"""

import argparse
import sys
import numpy as np
from pyscf import gto, scf
from pyscf.gto import intor_cross
import os

# Make the sibling modules (bond_symmetry, basis modules, minpop_json_csv, ...)
# importable whether this file runs as a script or as part of the minpop package.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(_HERE, "basis_sets")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import minpop_rohf as R          # all MinPop machinery and printers
import minpop_chk as _chk        # checkpoint files: write, read, reproject
try:
    from gvbpp import GVBPP
    from gvbpp.pairing import pair_guess, occupation_blocks
    from gvbpp.gvbpp import _expm_antisym
except ImportError:              # reprojecting a checkpoint needs no gvbpp
    GVBPP = pair_guess = occupation_blocks = _expm_antisym = None


def _need_gvbpp():
    if GVBPP is None:
        raise R.MinPopSCFError(
            "the gvbpp package is not installed (github.com/keithgroup/gvbpp; "
            "pip install -e gvbpp). It is needed to run GVB-PP; reprojecting "
            "a stored GVB-PP wavefunction with -read-chk works without it.")


# ---------------------------------------------------------------------------
# Valence window for the pair search
# ---------------------------------------------------------------------------

def _chemical_core_orbitals(Z):
    """Doubly occupied orbitals to keep out of the GVB pair search: the
    closed-shell core below the valence shell, semicore included.
    gvbpp's default ([Ne] for 3d metals) leaves 3s3p in the search and the
    localized guess then pairs a 3p instead of a bond."""
    if Z <= 2:
        return 0
    if Z <= 10:
        return 1            # [He]
    if Z <= 18:
        return 5            # [Ne]
    if Z <= 36:
        return 9            # [Ar]          3d / 4s4p valence
    if Z <= 54:
        return 18           # [Kr]          4d / 5s5p
    if Z <= 71:
        return 27           # [Xe]          4f 5d 6s (lanthanides)
    if Z <= 80:
        return 34           # [Xe]4f14      5d 6s
    if Z <= 86:
        return 39           # [Xe]4f14 5d10 6s6p
    if Z <= 103:
        return 43           # [Rn]          5f 6d 7s (actinides)
    return 50               # [Rn]5f14


def valence_window(mol, mf, n_pairs):
    """Number of highest doubly occupied orbitals to search for pairs."""
    docc, _, _ = occupation_blocks(mf)
    n_core = sum(_chemical_core_orbitals(int(mol.atom_charge(i))) for i in range(mol.natm))
    return max(n_pairs, len(docc) - n_core)


def run_gvbpp_multistart(mol, mf, n_pairs, window, guess='auto', n_trials=2,
                         seed=0, verbose=0, mo_start=None):
    """gvbpp's multi-start, with the localized pair search restricted to
    `window` highest doubly occupied orbitals. Returns the lowest-energy
    converged GVBPP object and the list of (name, energy, converged).

    mo_start : starting orbitals taken from a checkpoint (-guess-chk). They
    run first, as the start named 'checkpoint'. guess='checkpoint' runs only
    that start; any other guess adds its own starts and the lowest energy
    wins. Without mo_start, guess='checkpoint' falls back to 'auto'."""
    starts = []
    if mo_start is not None:
        starts.append(('checkpoint', mo_start))
    elif guess == 'checkpoint':
        guess = 'auto'
    if guess in ('auto', 'localized'):
        mo_l, _ = pair_guess(mf, n_pairs, method='localized', window=window)
        starts.append(('localized', mo_l))
    if guess in ('auto', 'canonical'):
        mo_c, _ = pair_guess(mf, n_pairs, method='canonical')
        starts.append(('canonical', mo_c))
    if guess == 'auto' and n_trials > 0:
        base = dict(starts)['localized']
        _, socc, _ = occupation_blocks(mf)
        ncas = 2 * n_pairs + len(socc)
        lo = len(occupation_blocks(mf)[0]) - n_pairs
        rng = np.random.default_rng(seed)
        for k in range(int(n_trials)):
            a = rng.normal(scale=0.35, size=(ncas, ncas))
            u = _expm_antisym(a - a.T)
            mo_t = base.copy()
            mo_t[:, lo:lo + ncas] = base[:, lo:lo + ncas] @ u
            starts.append((f'perturbed{k}', mo_t))
    best, log = None, []
    for name, mo in starts:
        g = GVBPP(mol, n_pairs)
        g.kernel(mf=mf, mo_coeff=mo, verbose=verbose)
        log.append((name, g.e_tot, bool(g.converged)))
        if g.converged and (best is None or g.e_tot < best.e_tot):
            best = g
    return best, log

__version__ = "1.1.0"


# ---------------------------------------------------------------------------
# Projection of fractionally occupied natural orbitals
# ---------------------------------------------------------------------------

def project_natural_orbitals(nos, occ, kinds, pair_of, S_cross, S_min,
                             keep_tol=0.5, verbose=True):
    """
    Project NOs onto the minimal basis with the representability rule.

    Parameters
    ----------
    nos : (nao, n) array of NOs in the computational basis
    occ : (n,) occupations
    kinds : list of 'core' | 'bond' | 'corr' | 'open'
    pair_of : list, pair index for 'bond'/'corr' NOs, -1 otherwise
    S_cross, S_min : overlap matrices (minimal x computational, minimal x minimal)
    keep_tol : minimum independent fraction for a correlating NO to be kept

    Returns
    -------
    C_min : (nmin, nkept) orthonormalized projected NOs
    occ_kept : (nkept,) occupations after folding dropped n_c into n_b
    kept_idx : indices into nos of the kept NOs
    report : list of (pair, kept(bool), independent_fraction)
    """
    S_min_inv = np.linalg.inv(S_min)
    C_proj = S_min_inv @ S_cross @ nos                   # raw projections
    order = np.argsort(-occ, kind='stable')              # decreasing occupation
    kept, occ_kept = [], occ.astype(float).copy()
    basis_kept = np.zeros((S_min.shape[0], 0))
    report = {}
    for i in order:
        v = C_proj[:, i]
        norm2 = float(v @ S_min @ v)                     # captured fraction of NO i
        if basis_kept.shape[1]:
            # remove components along already kept projected NOs (S_min metric)
            G = basis_kept.T @ S_min @ basis_kept
            coef = np.linalg.solve(G, basis_kept.T @ S_min @ v)
            resid = v - basis_kept @ coef
        else:
            resid = v
        indep = float(resid @ S_min @ resid)             # independent fraction
        if kinds[i] == 'corr':
            report[pair_of[i]] = (indep >= keep_tol, indep, norm2)
            if indep < keep_tol:
                # fold n_c into the bonding partner
                b = [k for k in range(len(kinds)) if kinds[k] == 'bond' and pair_of[k] == pair_of[i]][0]
                occ_kept[b] += occ_kept[i]
                occ_kept[i] = 0.0
                continue
        elif indep < keep_tol:
            raise RuntimeError(f"{kinds[i]} NO #{i + 1} has only {100 * indep:.1f}% of its "
                               f"norm independent in the minimal basis; the minimal basis "
                               f"cannot represent this wavefunction")
        kept.append(i)
        basis_kept = np.column_stack([basis_kept, v])
    kept = sorted(kept)
    C = C_proj[:, kept]
    M = C.T @ S_min @ C
    w, U = np.linalg.eigh(M)
    C_min = C @ (U @ np.diag(1.0 / np.sqrt(w)) @ U.T)   # symmetric orthonormalization
    return C_min, occ_kept[kept], kept, report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def minpop_gvbpp(g, minimal_basis='sto-3g', keep_tol=0.5, azimuthal_gauge=True,
                 verbose=True):
    """MinPop analysis of a converged gvbpp.GVBPP object `g`."""
    mol = g.mol
    mol_min = R._build_minimal_basis_mol(mol, minimal_basis=minimal_basis)
    if verbose:
        tag = {'minao': 'MINAO',
               'anorccmb': 'ANO-RCC-MB (all-electron, generic (n,l,m) order)',
               'anorccmbocc': 'ANO-RCC-MB-OCC (all-electron, no empty valence p beyond Ar)',
               'sto3gano': f'STO-3G on Z<={R.MBS_LIGHT_MAX_Z}, ANO-RCC-MB on Z>{R.MBS_LIGHT_MAX_Z}',
               'sto3ganoocc': f'STO-3G on Z<={R.MBS_LIGHT_MAX_Z}, ANO-RCC-MB-OCC on Z>{R.MBS_LIGHT_MAX_Z}',
               }.get(R._mbs_key(minimal_basis), 'STO-3G (Gaussian MBS convention)')
        if '+' in str(minimal_basis):
            a, b = [s.strip() for s in minimal_basis.split('+')]
            tag = f"{a.upper()} on Z<={R.MBS_LIGHT_MAX_Z}, {b.upper()} on Z>{R.MBS_LIGHT_MAX_Z}"
        print(f" Minimal-basis projection: {tag}")

    S_cross = intor_cross('int1e_ovlp', mol_min, mol)
    S_min = mol_min.intor('int1e_ovlp')

    # --- assemble NOs: core | b1 c1 b2 c2 ... | open ---
    ncore, npair, nopen = g.ncore, g.n_pairs, g.n_open
    nos = g.mo_coeff[:, :ncore + 2 * npair + nopen]
    occ = np.concatenate([np.full(ncore, 2.0), g.occupations[:2 * npair],
                          np.ones(nopen)])
    kinds = ['core'] * ncore + ['bond', 'corr'] * npair + ['open'] * nopen
    pair_of = [-1] * ncore + [p for p in range(npair) for _ in (0, 1)] + [-1] * nopen

    C_min, occ_kept, kept, report = project_natural_orbitals(
        nos, occ, kinds, pair_of, S_cross, S_min, keep_tol=keep_tol)
    kinds_kept = [kinds[i] for i in kept]
    if verbose:
        # Occupation-weighted share of the GVB-PP density inside the minimal-basis
        # span, in the same line format as the HF/KS drivers so it parses alike.
        C_raw = np.linalg.solve(S_min, S_cross @ nos)
        frac = np.einsum('ai,ab,bi->i', C_raw, S_min, C_raw)       # captured fraction per NO
        n_el = float(occ.sum())
        captured = float((occ * frac).sum())
        strong = np.where(occ >= 0.5)[0]                           # core, bonding, open
        worst = strong[np.argmin(frac[strong])]
        leak = n_el - captured
        print(f" Projection completeness (natural orbitals, occupation-weighted): "
              f"{captured:.5f} of {n_el:.0f} electrons captured, leak {leak:.5f} "
              f"({100.0 * leak / n_el:.3f}%); least-captured MO #{worst + 1} at "
              f"{100.0 * frac[worst]:.2f}%, {int((frac[strong] < 0.99).sum())} MOs below 99%")
    pair_kept = [pair_of[i] for i in kept]

    # --- densities: pairs and core split evenly, open shells alpha only ---
    w_alpha = np.array([1.0 if k == 'open' else o / 2 for k, o in zip(kinds_kept, occ_kept)])
    w_beta = np.array([0.0 if k == 'open' else o / 2 for k, o in zip(kinds_kept, occ_kept)])
    dm_alpha = (C_min * w_alpha) @ C_min.T
    dm_beta = (C_min * w_beta) @ C_min.T
    dm_total = dm_alpha + dm_beta
    dm_spin = dm_alpha - dm_beta

    # --- per-pair contribution to the MBS bond population (PySCF AO order) ---
    pair_rows = []
    try:
        from bond_symmetry import bond_components
    except ImportError:
        bond_components = None
    at = np.array([l[0] for l in mol_min.ao_labels(fmt=None)])
    for p in range(npair):
        cols = [k for k, (kd, pp) in enumerate(zip(kinds_kept, pair_kept)) if pp == p]
        dm_p = (C_min[:, cols] * occ_kept[cols]) @ C_min[:, cols].T
        pop_p = dm_p * S_min
        gross_atom = np.array([pop_p[at == a].sum() for a in range(mol.natm)])
        a1, a2 = np.argsort(-gross_atom)[:2]
        i, j = sorted((int(a1), int(a2)))
        lone = gross_atom[a2] < 0.1          # pair lives on one atom: lone pair
        bond_pop = pop_p[np.ix_(at == i, at == j)].sum()
        cls = ''
        if lone:
            i = j = int(a1)
            bond_pop = 0.0
        elif bond_components is not None and abs(bond_pop) > 1e-6:
            rec = bond_components(mol_min, dm_p, S_min, pairs=[(i, j)])[(i, j)]
            cls = max(('sigma', 'pi', 'delta', 'phi'), key=lambda c: abs(rec[c]))
            cls = {'sigma': '\u03c3', 'pi': '\u03c0', 'delta': '\u03b4', 'phi': '\u03c6'}[cls]
        n_b, n_c = g.occupations[2 * p], g.occupations[2 * p + 1]
        kept_c, indep, norm2 = report.get(p, (False, 0.0, 0.0))
        pair_rows.append((p + 1, n_b, n_c, g.overlaps[p], kept_c, indep, i, j,
                          gross_atom[i], gross_atom[j], bond_pop, cls))

    # --- regular MinPop pipeline from here on ---
    pop_alpha = R._mulliken_pop_matrix(dm_alpha, S_min)
    pop_beta = R._mulliken_pop_matrix(dm_beta, S_min)
    pop_total = R._mulliken_pop_matrix(dm_total, S_min)
    pop_spin = R._mulliken_pop_matrix(dm_spin, S_min)

    reorder, ao_labels_raw = R._get_gaussian_ao_order(mol_min, minimal_basis)
    dm_alpha = R._reorder_matrix(dm_alpha, reorder)
    dm_beta = R._reorder_matrix(dm_beta, reorder)
    dm_total = R._reorder_matrix(dm_total, reorder)
    dm_spin = R._reorder_matrix(dm_spin, reorder)
    pop_alpha = R._reorder_matrix(pop_alpha, reorder)
    pop_beta = R._reorder_matrix(pop_beta, reorder)
    pop_total = R._reorder_matrix(pop_total, reorder)
    pop_spin = R._reorder_matrix(pop_spin, reorder)
    ao_labels = [R._convert_label_to_gaussian(lbl) for lbl in ao_labels_raw]

    if azimuthal_gauge:
        dm_list, theta = R._apply_azimuthal_gauge(
            [dm_alpha, dm_beta, dm_total, dm_spin], ao_labels,
            mol.atom_coords(), verbose=verbose)
    else:
        dm_list, theta = [dm_alpha, dm_beta, dm_total, dm_spin], None
    if theta is not None:
        dm_alpha, dm_beta, dm_total, dm_spin = dm_list
        S_g = R._reorder_matrix(S_min, reorder)
        pop_alpha = R._mulliken_pop_matrix(dm_alpha, S_g)
        pop_beta = R._mulliken_pop_matrix(dm_beta, S_g)
        pop_total = R._mulliken_pop_matrix(dm_total, S_g)
        pop_spin = R._mulliken_pop_matrix(dm_spin, S_g)

    gross = np.column_stack([pop_total.sum(0), pop_alpha.sum(0), pop_beta.sum(0),
                             pop_alpha.sum(0) - pop_beta.sum(0)])
    condensed = R._condense_to_atoms(pop_total, ao_labels)
    spin_atomic = R._condense_to_atoms(pop_spin, ao_labels)
    nuclear = np.array([mol.atom_charge(i) for i in range(mol.natm)])
    results = {
        'dm_alpha': dm_alpha, 'dm_beta': dm_beta, 'dm_total': dm_total,
        'dm_spin': dm_spin, 'pop_total': pop_total, 'pop_spin': pop_spin,
        'gross_orbital_pop': gross, 'condensed_to_atoms': condensed,
        'spin_atomic': spin_atomic,
        'mulliken_charges': nuclear - condensed.sum(1),
        'spin_populations': spin_atomic.sum(1),
        'ao_labels': ao_labels, 'ao_labels_raw': ao_labels_raw,
        'ao_reorder': reorder,
        'gvb_pairs': pair_rows, 'gvb_e_tot': g.e_tot, 'gvb_e_corr': g.e_corr,
        'no_occupations_kept': occ_kept, 'no_kinds_kept': kinds_kept,
    }
    if verbose:
        _print_results(results, mol_min, ncore, npair, nopen, len(kept),
                       ncore + 2 * npair + nopen)
    return results


def _print_results(results, mol_min, ncore, npair, nopen, nkept, ntotal):
    ao_labels = results['ao_labels']
    print(f"GVB-PP orbital structure: {ncore} core, {npair} pairs, {nopen} open-shell "
          f"orbitals; {nkept} of {ntotal} natural orbitals projected")
    print("=" * 60)
    print("MinPop Analysis (GVB-PP)")
    print("=" * 60)
    print(f" GVB-PP energy = {results['gvb_e_tot']:.10f}   "
          f"E_corr = {results['gvb_e_corr']:.6f}")
    print()
    print(" GVB-PP pairs:")
    print("  pair     n_bond     n_corr      S_ab  corr NO     pair on       gross(A)   gross(B)  "
          "MBS bond pop   sym")
    warn_core = []
    for (p, nb, nc, sab, keptc, indep, i, j, ga, gb, bp, cls) in results['gvb_pairs']:
        kc = f"kept {100 * indep:3.0f}%" if keptc else f"merged {100 * indep:3.0f}%"
        if i == j:
            core_like = nc < 0.005
            name = f"{mol_min.atom_symbol(i)}{i + 1} " + ("core-like" if core_like else "lone pair")
            if core_like:
                warn_core.append(p)
            print(f"  {p:4d}  {nb:9.5f}  {nc:9.5f}  {sab:8.4f}  {kc:11s} {name:14s}"
                  f"{ga:9.5f}")
        else:
            name = f"{mol_min.atom_symbol(i)}{i + 1}-{mol_min.atom_symbol(j)}{j + 1}"
            print(f"  {p:4d}  {nb:9.5f}  {nc:9.5f}  {sab:8.4f}  {kc:11s} {name:14s}"
                  f"{ga:9.5f}  {gb:9.5f}  {bp:12.6f}   {cls}")
    print("   corr NO: whether the correlating natural orbital of the pair is")
    print("   representable in the minimal basis (independent fraction); when")
    print("   merged, its occupation is added to the bonding orbital.")
    if warn_core:
        print(f"   WARNING: pair(s) {warn_core} are doubly occupied orbitals localized on one "
              f"atom with negligible correlation (n_corr < 0.005): a semicore orbital was "
              f"paired and a bonding orbital probably stayed frozen in the core. Narrow "
              f"-gvb-window or check the active space.")
    print()
    R._print_density_matrix(results['dm_alpha'], ao_labels, "Alpha  MBS Density Matrix")
    R._print_density_matrix(results['dm_beta'], ao_labels, "Beta  MBS Density Matrix")
    R._print_density_matrix(results['pop_total'], ao_labels,
                            "Full MBS Mulliken population analysis", prefix="    ")
    R._print_gross_populations(results['gross_orbital_pop'], ao_labels)
    R._print_atomic_matrix(results['condensed_to_atoms'], mol_min,
                           "Condensed to atoms (all electrons)")
    R._print_atomic_matrix(results['spin_atomic'], mol_min,
                           "Atomic-Atomic Spin Densities")
    R._print_mulliken_summary(results['mulliken_charges'],
                              results['spin_populations'], mol_min)
    print("=" * 60)
    if R.BOND_SYMMETRY:
        R._print_symmetry_section(results, mol_min, shells=R.BOND_SYMMETRY_SHELLS)
        print("=" * 60)


def _gvb_from_xyz(xyz_file, n_pairs, charge=0, multiplicity=1, basis='6-31+G',
                       ecp=None, verbose=True, basis_dir=None,
                       standard_orientation=True, azimuthal_gauge=True,
                       guess='minao', max_cycle=128, soscf=False, x2c=False,
                       minimal_basis='sto-3g', keep_tol=0.5,
                       gvb_guess='auto', gvb_trials=2, gvb_verbose=0, gvb_window=None,
                       gvb_seed=0, basis_light=None, light_max_z=18, x2c_atomic=False,
                       guess_chk=None):
    """Reference RHF/ROHF and GVB-PP optimization from an XYZ file. Returns
    the converged gvbpp.GVBPP object (its .mf is the reference)."""
    _need_gvbpp()
    if n_pairs is None:
        raise ValueError("n_pairs is required to run GVB-PP")
    atom_str = R._read_xyz(xyz_file)
    orient_table = None
    if standard_orientation:
        atom_str, orient_table = R._standardize_atom_str(atom_str, search_dir=basis_dir)
    basis_obj, auto_ecp, basis_name = R._compose_basis(
        basis, basis_light, atom_str, light_max_z=light_max_z, search_dir=basis_dir)
    if ecp is None and auto_ecp:
        ecp = auto_ecp
        if verbose:
            print(f"Auto-detected ECP for heavy elements: {ecp}")
    mol = gto.M(atom=atom_str, basis=basis_obj, charge=charge,
                spin=multiplicity - 1, ecp=ecp, cart=False)
    if verbose:
        print(f"Molecule: {xyz_file}")
        print(f"Charge: {charge}, Multiplicity: {multiplicity}")
        print(f"Basis: {basis_name} (5D/7F spherical)")
        print(f"Atoms: {mol.natm}, Electrons: {mol.nelectron}")
        if orient_table:
            print()
            print(orient_table)
        print()

    mf = scf.ROHF(mol) if mol.spin else scf.RHF(mol)
    if x2c_atomic:
        x2c = True            # -x2c-atomic is a form of -x2c
    if x2c:
        mf = mf.x2c()
        if x2c_atomic:
            # atomic (DLU) decoupling: one X2C block per atom, cost linear in
            # the number of atoms; light atoms contribute negligibly
            mf.with_x2c.approx = 'atom1e'
        if verbose:
            print("Hamiltonian: spin-free X2C-1e (scalar relativistic)"
                  + (", atomic (atom-by-atom) decoupling" if x2c_atomic else ""))
    elif ecp is None and max(mol.atom_charge(i) for i in range(mol.natm)) > 36 and verbose:
        print("WARNING: all-electron basis on an atom beyond Kr without -x2c; "
              "non-relativistic orbitals for such atoms are not physical.")
    mf.max_cycle = max_cycle
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-6
    mf.diis_space = 8
    if soscf:
        if verbose:
            print("SCF: second-order (Newton) solver from the start (-soscf)")
        conv, mx = mf.conv_tol, mf.max_cycle
        mf = mf.newton()
        mf.conv_tol, mf.max_cycle = conv, mx
    if guess_chk:
        # start the reference from a stored wavefunction (any basis or frame)
        dm0 = _chk.guess_density(guess_chk, mol, "ROHF" if mol.spin else "RHF",
                                 verbose=verbose)
    else:
        dm0 = mf.get_init_guess(key=guess) if guess != 'minao' else None
    mf.kernel(dm0=dm0)
    R._require_converged(mf, "SCF")
    # stored GVB-PP natural orbitals as the first (or only) GVB-PP start
    mo_start = (_chk.gvb_start_orbitals(guess_chk, mol, mf, n_pairs, verbose=verbose)
                if guess_chk else None)
    if mo_start is None and gvb_guess == 'checkpoint':
        gvb_guess = 'auto'
    if verbose:
        print(f"GVB-PP: {n_pairs} pairs, guess={gvb_guess}, trials={gvb_trials}")

    window = valence_window(mol, mf, n_pairs) if gvb_window is None else gvb_window
    if verbose:
        print(f"GVB-PP pair search window: {window} highest doubly occupied orbitals "
              f"(semicore excluded)")
    g, log = run_gvbpp_multistart(mol, mf, n_pairs, window, guess=gvb_guess, seed=gvb_seed,
                                  n_trials=gvb_trials, verbose=gvb_verbose,
                                  mo_start=mo_start)
    if verbose:
        for name, e, conv in log:
            print(f"  guess={name:<11s} E = {e:.10f}  converged = {conv}")
    if g is None:
        raise R.MinPopSCFError("GVB-PP orbital optimization did not converge from any start")
    if verbose:
        print(f"converged GVB-PP energy = {g.e_tot:.10f}")
        print()
    return g


def run_gvbpp_from_xyz(xyz_file=None, n_pairs=None, *args, chk=None,
                       read_chk=None, explicit=None, **kwargs):
    """
    Run the reference SCF, GVB-PP and the MinPop analysis from an XYZ file, or
    reproject a stored GVB-PP wavefunction. Takes every argument of
    _gvb_from_xyz (charge, multiplicity, basis, minimal_basis, keep_tol,
    gvb_guess, guess_chk, ...), plus:

    chk : str, optional
        Write the converged GVB-PP wavefunction (natural orbitals, pair
        occupations, energies, and the RHF/ROHF reference) to this PySCF
        checkpoint file.
    read_chk : str, optional
        Skip the SCF and the GVB-PP optimization and project the wavefunction
        stored in this checkpoint. xyz_file and n_pairs are then optional;
        only minimal_basis, keep_tol, azimuthal_gauge and verbose still
        matter. The gvbpp package is not needed for this.
    explicit : dict, optional
        Settings the caller insists on (command line only); a checkpoint that
        disagrees with them is refused.

    guess_chk (a _gvb_from_xyz argument) starts the run from a checkpoint
    instead: the reference SCF from its SCF orbitals and, when the file holds
    a GVB-PP wavefunction with the same pair count, the GVB-PP optimization
    from its natural orbitals (gvb_guess='checkpoint' to use only that start).
    """
    import inspect
    if xyz_file is None and not read_chk:
        raise ValueError("xyz_file is required unless read_chk is given")
    bound = inspect.signature(_gvb_from_xyz).bind(xyz_file, n_pairs, *args, **kwargs)
    bound.apply_defaults()
    kw = dict(bound.arguments)
    return _chk.run_with_checkpoint(
        "GVBPP", _gvb_from_xyz, minpop_gvbpp, kw,
        dict(minimal_basis=kw["minimal_basis"], keep_tol=kw["keep_tol"],
             azimuthal_gauge=kw["azimuthal_gauge"], verbose=kw["verbose"]),
        chk=chk, read_chk=read_chk, explicit=explicit, read_xyz=R._read_xyz)


def main():
    p = argparse.ArgumentParser(
        description="MinPop on a GVB-PP wavefunction (keithgroup/gvbpp)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python minpop_gvbpp.py -xyz pt2.xyz -mult 3 -npairs 2 "
               "-basis ano-rcc-vdzp -minimal-basis ano-rcc-mb -x2c -symmetry\n"
               "  python minpop_gvbpp.py -xyz cr2.xyz -npairs 6 -basis ano-rcc-vdzp -x2c "
               "-gvb-trials 20 -chk cr2.chk          # keep the GVB-PP wavefunction\n"
               "  python minpop_gvbpp.py -read-chk cr2.chk -minimal-basis ano-rcc-mb-occ "
               "-symmetry   # reproject, no SCF, no GVB-PP\n"
               "  python minpop_gvbpp.py -xyz cr2.xyz -npairs 6 -basis ano-rcc-vtzp -x2c "
               "-guess-chk cr2.chk   # GVB-PP from the stored orbitals")
    p.add_argument("-xyz", dest="xyz_file", default=None,
                   help="Path to XYZ geometry file (required unless -read-chk is given)")
    p.add_argument("-npairs", type=int, dest="n_pairs", default=None,
                   help="Number of GVB pairs to correlate (open shells excluded). "
                        "Required unless -read-chk is given; the pair count then "
                        "comes from the checkpoint")
    p.add_argument("-charge", type=int, default=0)
    p.add_argument("-mult", type=int, default=1)
    p.add_argument("-basis", default="6-31+G")
    p.add_argument("-basis-light", dest="basis_light", default=None,
                   help="Basis for atoms up to -light-max-z (default Ar); -basis then "
                        "applies to the heavier atoms only")
    p.add_argument("-light-max-z", dest="light_max_z", type=int, default=18)
    p.add_argument("-basis-dir", dest="basis_dir", default=None)
    p.add_argument("-minimal-basis", "-mbs", dest="minimal_basis",
                   type=R._check_minimal_basis, default="sto-3g",
                   help="sto-3g, minao, ano-rcc-mb, ano-rcc-mb-occ, sto-3g-ano, or LIGHT+HEAVY")
    p.add_argument("-no-keep", dest="keep_tol", type=float, default=0.5,
                   help="Minimum independent fraction for a correlating NO to be "
                        "projected rather than merged into its bonding NO (default 0.5)")
    p.add_argument("-label-style", dest="label_style",
                   choices=["gaussian", "physical"], default=None)
    p.add_argument("-symmetry", "-sigma-pi", dest="symmetry", action="store_true", default=False,
                   help="σ/π/δ/φ resolution of the bond populations, condensed to atoms")
    p.add_argument("-symmetry-shells", "-sigma-pi-shells", dest="symmetry_shells", action="store_true",
                   default=False, help="With -symmetry: also condensed to shells")
    p.add_argument("-x2c", dest="x2c", action="store_true", default=False)
    p.add_argument("-x2c-atomic", dest="x2c_atomic", action="store_true", default=False,
                   help="X2C with atom-by-atom (DLU) decoupling, linear-scaling (implies -x2c)")
    p.add_argument("-ecp", default=None)
    p.add_argument("-no-azimuthal-gauge", dest="azimuthal_gauge", action="store_false",
                   default=True)
    p.add_argument("-no-std-orient", dest="standard_orientation", action="store_false",
                   default=True)
    p.add_argument("-soscf", dest="soscf", action="store_true", default=False)
    p.add_argument("-max-cycle", dest="max_cycle", type=int, default=128)
    p.add_argument("-guess", dest="guess", default="minao",
                   choices=["minao", "huckel", "vsap", "sap", "atom", "1e"])
    p.add_argument("-gvb-guess", dest="gvb_guess", default="auto",
                   choices=["auto", "localized", "canonical", "checkpoint"],
                   help="gvbpp starting-orbital strategy (default auto: multi-start). "
                        "With -guess-chk on a GVB-PP checkpoint the default becomes "
                        "checkpoint: the stored natural orbitals are the only start. "
                        "Give -gvb-guess or -gvb-trials to run those starts as well; "
                        "the lowest energy wins")
    p.add_argument("-gvb-trials", dest="gvb_trials", type=int, default=2)
    p.add_argument("-gvb-seed", dest="gvb_seed", type=int, default=0,
                   help="Seed for the perturbed starts (default 0); a different seed gives "
                        "an independent family of starting rotations")
    p.add_argument("-gvb-window", dest="gvb_window", type=int, default=None,
                   help="Highest doubly occupied orbitals searched for pairs "
                        "(default: valence orbitals only, semicore excluded)")
    p.add_argument("-gvb-verbose", dest="gvb_verbose", type=int, default=0)
    g = p.add_argument_group("checkpoint")
    g.add_argument("-chk", dest="chk", metavar="PATH", default=None,
                   help="Write the converged GVB-PP wavefunction (molecule, basis, "
                        "natural orbitals, pair occupations, energies, and the "
                        "RHF/ROHF reference) to this PySCF checkpoint file. Default: "
                        "<xyz name>_gvbpp.chk in the current directory. Give a name "
                        "whenever runs of the same molecule share a directory")
    g.add_argument("-no-chk", dest="no_chk", action="store_true", default=False,
                   help="Do not write a checkpoint file")
    g.add_argument("-read-chk", dest="read_chk", metavar="PATH", default=None,
                   help="Skip the SCF and the GVB-PP optimization: read the GVB-PP "
                        "wavefunction from this checkpoint and project it. -xyz and "
                        "-npairs are then optional; basis, charge, multiplicity, "
                        "Hamiltonian and pair count come from the file, and a flag "
                        "that disagrees with it stops the run. -minimal-basis, "
                        "-no-keep, -light-max-z, -label-style, -symmetry, "
                        "-symmetry-shells, -no-azimuthal-gauge, -json and -csv-long "
                        "apply as usual. Does not need the gvbpp package")
    g.add_argument("-guess-chk", dest="guess_chk", metavar="PATH", default=None,
                   help="Run the SCF and GVB-PP, started from this checkpoint. The "
                        "reference SCF starts from its SCF orbitals (any MinPop or "
                        "PySCF checkpoint). When it is a GVB-PP checkpoint with the "
                        "same pair count, the GVB-PP optimization starts from its "
                        "natural orbitals, which then replace the multi-start (see "
                        "-gvb-guess). Basis and frame may differ; atoms, charge and "
                        "multiplicity must match. Replaces -guess")
    p.add_argument("-json", dest="json_path", metavar="PATH", default=None,
                   help="Also serialize this run to a JSON record (schema 1.4) via "
                        "minpop_json_csv.py; a failed export never fails the run")
    p.add_argument("-json-xyz-dir", dest="json_xyz_dir", action="append", metavar="DIR",
                   default=None, help="Directory with the input .xyz (repeatable)")
    p.add_argument("-csv-long", dest="csv_long", metavar="PATH", default=None,
                   help="Also write the tidy sparse CSV of every printed quantity")
    p.add_argument("-csv-sparse-tol", dest="csv_sparse_tol", type=float, default=0.0)
    args = p.parse_args()
    chk_path, read_chk, chk_explicit, scf_xyz = _chk.resolve_cli(p, args, "GVBPP")
    if not read_chk and args.n_pairs is None:
        p.error("-npairs N is required to run GVB-PP (only -read-chk takes it "
                "from the checkpoint)")
    # a GVB-PP checkpoint as the start replaces the multi-start unless the
    # user asks for those starts too
    if (args.guess_chk and not read_chk and "gvb_guess" not in chk_explicit
            and "gvb_trials" not in chk_explicit):
        args.gvb_guess = "checkpoint"

    R.LABEL_STYLE = args.label_style
    R.MBS_LIGHT_MAX_Z = args.light_max_z
    R.BOND_SYMMETRY = args.symmetry or args.symmetry_shells
    R.BOND_SYMMETRY_SHELLS = args.symmetry_shells
    import contextlib
    exporting = bool(args.json_path or args.csv_long)
    ctx = R._capture_report() if exporting else contextlib.nullcontext()
    failure = None
    with ctx as buf:
        print("Command line: python " + " ".join(sys.argv))
        try:
            run_gvbpp_from_xyz(scf_xyz, args.n_pairs, chk=chk_path, read_chk=read_chk,
                               explicit=chk_explicit, guess_chk=args.guess_chk,
                               charge=args.charge,
                               multiplicity=args.mult, basis=args.basis, ecp=args.ecp,
                               basis_dir=args.basis_dir,
                               standard_orientation=args.standard_orientation,
                               azimuthal_gauge=args.azimuthal_gauge, guess=args.guess,
                               max_cycle=args.max_cycle, soscf=args.soscf, x2c=args.x2c,
                               minimal_basis=args.minimal_basis, keep_tol=args.keep_tol,
                               gvb_guess=args.gvb_guess, gvb_trials=args.gvb_trials,
                               gvb_verbose=args.gvb_verbose, gvb_window=args.gvb_window,
                               gvb_seed=args.gvb_seed, basis_light=args.basis_light,
                               light_max_z=args.light_max_z, x2c_atomic=args.x2c_atomic)
        except (R.MinPopSCFError, _chk.CheckpointError) as e:
            print(f"MinPop ERROR: {e}")
            failure = e
        text = buf.getvalue() if exporting else None
    if exporting and failure is None:
        R._export_outputs(text, args)
    if failure is not None:
        sys.exit(1)

if __name__ == "__main__":
    main()
