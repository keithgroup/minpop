#!/usr/bin/env python3
"""
MinPop checkpoint files: keep the converged SCF, reproject it later.

The SCF is the expensive part of a MinPop run and the projection is cheap. The
drivers (minpop_rohf / minpop_uhf / minpop_rks / minpop_uks, and minpop_gvbpp
for GVB-PP wavefunctions) therefore obtain their wavefunction through this
module:

    run_with_checkpoint()   run the SCF and write a checkpoint, or read a
                            checkpoint and skip the SCF; then run the analysis
    save() / restore()      write / read one checkpoint
    guess_density()         a checkpoint as the initial guess of a new SCF
    gvb_start_orbitals()    stored GVB-PP orbitals as the start of a new GVB-PP
    restore_gvbpp()         a stored GVB-PP wavefunction, for reprojection
    main()                  `minpop-chk FILE ...`: reproject a checkpoint with
                            whichever driver wrote it, or print what it holds

A checkpoint is an ordinary PySCF chkfile (HDF5): the molecule with its
computational basis and ECP, and the SCF record (e_tot, mo_energy, mo_occ,
mo_coeff). MinPop adds one group, `minpop`, with the method, the settings that
define the SCF, and the SCF section of the report. The orbitals stored are the
final ones the analysis used (after -stable and the Newton tightening), in the
orientation the analysis used.

A reprojection prints the stored SCF section verbatim, so the report and the
JSON record of a reprojection match those of the original run apart from the
command line, the checkpoint line, and everything the minimal basis changes.

Plain PySCF chkfiles written outside MinPop (mf.chkfile = "x.chk") can be read
too. They carry no `minpop` group, so the header is rebuilt from the molecule
and from the flags given (-xc, -x2c, -basis are then labels only), and the
geometry is used in the frame it was stored in.

Command line
------------
    minpop-uks -xyz rh.xyz -mult 2 -basis def2-tzvpp -chk rh.chk
    minpop-uks -read-chk rh.chk -minimal-basis minao -symmetry -json rh_minao.json
    minpop-chk rh.chk -info
    minpop-chk rh.chk -minimal-basis sto-3g-ano-occ -symmetry-shells
    minpop-uks -xyz rh.xyz -mult 2 -basis def2-qzvpp -guess-chk rh.chk -chk rh_qz.chk

-read-chk runs no SCF: it projects the stored orbitals. -guess-chk runs the
SCF and only starts it from the stored orbitals (guess_density()).
"""

import argparse
import contextlib
import datetime
import io
import os
import re
import shlex
import sys

import numpy as np

__version__ = "1.2.0"

FORMAT_VERSION = 1
GROUP = "minpop"
GVB_GROUP = "gvbpp"

# method tag -> (unrestricted orbitals?, Kohn-Sham?, driver module, console script)
METHODS = {
    "ROHF": (False, False, "minpop_rohf", "minpop-rohf"),
    "UHF":  (True,  False, "minpop_uhf",  "minpop-uhf"),
    "RKS":  (False, True,  "minpop_rks",  "minpop-rks"),
    "UKS":  (True,  True,  "minpop_uks",  "minpop-uks"),
    # GVB-PP: the scf record is the RHF/ROHF reference; the GVB-PP natural
    # orbitals and occupations sit in their own group (GVB_GROUP)
    "GVBPP": (False, False, "minpop_gvbpp", "minpop-gvbpp"),
}

# Flags that define the SCF. With -read-chk they come from the checkpoint; one
# given explicitly must agree with it (a guard against a stale checkpoint).
SCF_DEFINING = ("charge", "mult", "basis", "basis_light", "cartesian", "x2c",
                "x2c_atomic", "xc", "grid_level", "ecp")
# Flags that steer how the SCF is reached. With -read-chk there is no SCF, so
# they do nothing; the run says so on stderr.
SCF_PROCEDURE = {
    "guess": "-guess", "guessmix": "-guessmix",
    "guessmix_angle": "-guessmix-angle", "guess_triplet": "-guess-triplet",
    "stable": "-stable", "stable_cycles": "-stable-cycles",
    "soscf": "-soscf", "newton_fallback": "-no-newton-fallback",
    "newton_conv_tol": "-newton-conv-tol", "max_cycle": "-max-cycle",
    "basis_dir": "-basis-dir", "standard_orientation": "-no-std-orient",
    "guess_chk": "-guess-chk",
    "gvb_guess": "-gvb-guess", "gvb_trials": "-gvb-trials",
    "gvb_seed": "-gvb-seed", "gvb_window": "-gvb-window",
    "gvb_verbose": "-gvb-verbose",
}


class CheckpointError(RuntimeError):
    """The checkpoint cannot be used for the requested analysis."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def default_path(xyz_file, method):
    """<xyz stem>_<method>.chk in the current directory, e.g. rhco3_uks.chk."""
    stem = os.path.splitext(os.path.basename(str(xyz_file)))[0] or "minpop"
    return f"{stem}_{method.lower()}.chk"


def _norm(name):
    return re.sub(r"[^a-z0-9*+]", "", str(name).lower())


def _text(v):
    """HDF5 hands strings back as bytes or numpy scalars."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _text(v.item())
    return v


def _plain(v):
    v = _text(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


@contextlib.contextmanager
def recording():
    """Tee stdout into a buffer while the SCF section of the report prints.

    PySCF binds lib.StreamObject.stdout at import time, so both it and
    sys.stdout are redirected (the same reason the JSON capture does both).
    """
    from pyscf import lib
    buf = io.StringIO()
    old_sys, old_lib = sys.stdout, lib.StreamObject.stdout
    tee = _Tee(old_sys, buf)
    sys.stdout = tee
    lib.StreamObject.stdout = tee
    try:
        yield buf
    finally:
        sys.stdout = old_sys
        lib.StreamObject.stdout = old_lib


def explicit_args(parser, argv=None):
    """{dest: value} for the options actually present on the command line."""
    saved = [(a, a.default, a.required) for a in parser._actions]
    saved_defaults = dict(parser._defaults)
    for a, _, _ in saved:
        a.default = argparse.SUPPRESS
        a.required = False
    parser._defaults.clear()
    try:
        ns, _ = parser.parse_known_args(argv)
    finally:
        for a, d, r in saved:
            a.default, a.required = d, r
        parser._defaults.update(saved_defaults)
    return vars(ns)


def _spin_square(mol, mo_coeff, mo_occ):
    from pyscf import scf
    ca, cb = mo_coeff
    oa, ob = mo_occ
    ss, s = scf.uhf.spin_square((ca[:, oa > 0], cb[:, ob > 0]),
                                mol.intor("int1e_ovlp"))
    return float(ss), float(s)


def _energy_line(e_tot, s2=None):
    """The line PySCF prints at convergence; the JSON export reads it."""
    line = f"converged SCF energy = {e_tot:.15g}"
    if s2 is not None:
        line += f"  <S^2> = {s2[0]:.8g}  2S+1 = {s2[1]:.8g}"
    return line


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
def save(mf, path, method, header="", settings=None, gvb=None):
    """Write the converged SCF in `mf` to `path`.

    method   : 'ROHF', 'UHF', 'RKS', 'UKS' or 'GVBPP'
    header   : the SCF section of the report as it was printed
    settings : the driver's keyword arguments (basis, xc, x2c, ...)
    gvb      : for 'GVBPP', the converged gvbpp.GVBPP object; `mf` is then its
               RHF/ROHF reference
    """
    from pyscf import lib
    from pyscf.scf import chkfile as scf_chk
    import pyscf
    if method not in METHODS:
        raise CheckpointError(f"unknown method {method!r}")
    s = dict(settings or {})
    mol = mf.mol
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if os.path.exists(path):
        os.remove(path)                       # never append to an older file
    scf_chk.dump_scf(mol, path, float(mf.e_tot), np.asarray(mf.mo_energy),
                     np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ),
                     overwrite_mol=True)
    grid = None
    if METHODS[method][1]:
        grid = s.get("grid_level")
        if grid is None:
            grid = getattr(getattr(mf, "grids", None), "level", None)
    x2c = bool(s.get("x2c") or s.get("x2c_atomic"))
    meta = {
        "format_version": FORMAT_VERSION,
        "method": method,
        "xc": str(s.get("xc")) if METHODS[method][1] else "",
        "grid_level": -1 if grid is None else int(grid),
        "basis": str(s.get("basis", "")),
        "basis_light": str(s.get("basis_light") or ""),
        "light_max_z": int(s.get("light_max_z", 18)),
        "cartesian": bool(mol.cart),
        "x2c": x2c,
        "x2c_atomic": bool(s.get("x2c_atomic")),
        "ecp": "" if not mol.ecp else str(mol.ecp),
        "charge": int(mol.charge),
        "multiplicity": int(mol.spin) + 1,
        "xyz_file": str(s.get("xyz_file") or ""),
        "standard_orientation": bool(s.get("standard_orientation", True)),
        "converged": bool(getattr(mf, "converged", True)),
        "guess_chk": str(s.get("guess_chk") or ""),
        "n_pairs": -1 if gvb is None else int(gvb.n_pairs),
        "header": str(header or ""),
        "command_line": "python " + " ".join(shlex.quote(a) for a in sys.argv),
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
                       .isoformat(timespec="seconds"),
        "pyscf_version": pyscf.__version__,
        "minpop_chk_version": __version__,
    }
    if gvb is not None:
        lib.chkfile.save(path, GVB_GROUP, {
            "e_tot": float(gvb.e_tot),
            "e_corr": float(gvb.e_corr),
            "e_ref": float(mf.e_tot),
            "mo_coeff": np.asarray(gvb.mo_coeff),      # core | b1 c1 b2 c2 .. | open | virtual
            "occupations": np.asarray(gvb.occupations),
            "overlaps": np.asarray(gvb.overlaps),
            "ncore": int(gvb.ncore),
            "n_pairs": int(gvb.n_pairs),
            "n_open": int(gvb.n_open),
            "converged": bool(gvb.converged),
        })
    lib.chkfile.save(path, GROUP, meta)
    return path


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def load(path):
    """(mol, scf record, meta) from a checkpoint; meta is None for a plain
    PySCF chkfile written outside MinPop."""
    from pyscf import lib
    from pyscf.scf import chkfile as scf_chk
    if not os.path.isfile(path):
        raise CheckpointError(f"checkpoint file not found: {path}")
    try:
        mol, rec = scf_chk.load_scf(path)
    except Exception as exc:
        raise CheckpointError(
            f"{path} is not a readable PySCF checkpoint "
            f"({type(exc).__name__}: {exc})")
    if rec is None or "mo_coeff" not in rec or "mo_occ" not in rec:
        raise CheckpointError(f"{path} holds no SCF orbitals (scf/mo_coeff, "
                              f"scf/mo_occ)")
    try:
        meta = lib.chkfile.load(path, GROUP)
    except Exception:
        meta = None
    if meta is not None:
        meta = {k: _plain(v) for k, v in meta.items()}
    return mol, rec, meta


def _unrestricted(rec):
    return np.asarray(rec["mo_coeff"]).ndim == 3


def stored_method(rec, meta):
    """The method tag in the checkpoint, or None for a plain PySCF chkfile."""
    return meta.get("method") if meta else None


def _synthetic_header(path, mol, rec, meta, method, labels):
    """Header for a checkpoint that carries no printed SCF section: a plain
    PySCF chkfile, or a MinPop one written by a -q run."""
    labels = labels or {}
    m = meta or {}
    basis = labels.get("basis") or m.get("basis") or (
        mol.basis if isinstance(mol.basis, str) else "custom")
    light = labels.get("basis_light") or m.get("basis_light")
    if light:
        z = m.get("light_max_z", 18)
        basis = f"{basis} on Z>{z}, {light} on Z<={z}"
    out = [f"Molecule: {m.get('xyz_file') or path}",
           f"Charge: {mol.charge}, Multiplicity: {mol.spin + 1}",
           f"Basis: {basis} "
           f"({'6D/10F Cartesian' if mol.cart else '5D/7F spherical'})",
           f"Atoms: {mol.natm}, Electrons: {mol.nelectron}", ""]
    x2c = labels.get("x2c", m.get("x2c"))
    atomic = labels.get("x2c_atomic", m.get("x2c_atomic"))
    if x2c or atomic:
        out.append("Hamiltonian: spin-free X2C-1e (scalar relativistic)"
                   + (", atomic (atom-by-atom) decoupling" if atomic else ""))
    if METHODS[method][1]:
        xc = labels.get("xc") or m.get("xc") or "unknown"
        grid = labels.get("grid_level")
        if grid is None:
            grid = m.get("grid_level", -1)
        grid = "unknown" if grid in (None, -1) else grid
        out.append(f"KS-DFT: {method} xc={xc} grid level={grid}")
    s2 = (_spin_square(mol, rec["mo_coeff"], rec["mo_occ"])
          if _unrestricted(rec) else None)
    out += [_energy_line(float(rec["e_tot"]), s2), ""]
    return "\n".join(out) + "\n"


def _check_wavefunction(path, mol, rec, method):
    unres, _, _, script = METHODS[method]
    c = np.asarray(rec["mo_coeff"])
    occ = np.asarray(rec["mo_occ"])
    if _unrestricted(rec) != unres:
        have = "unrestricted (separate alpha and beta)" if _unrestricted(rec) \
            else "restricted"
        want = [k for k, v in METHODS.items() if v[0] == _unrestricted(rec)]
        raise CheckpointError(
            f"{path} holds {have} orbitals, which {script} cannot analyse. "
            f"Use {' or '.join(METHODS[k][3] for k in want)}, or "
            f"`minpop-chk {path}`, which picks the driver itself.")
    nao = mol.nao_nr()
    if c.shape[-2] != nao:
        raise CheckpointError(
            f"{path}: orbitals have {c.shape[-2]} basis functions but the "
            f"stored molecule has {nao}")
    nelec = float(occ.sum())
    if abs(nelec - mol.nelectron) > 1e-6:
        raise CheckpointError(
            f"{path}: occupations sum to {nelec:g} electrons but the stored "
            f"molecule has {mol.nelectron}")
    if method == "RKS" and (mol.spin != 0 or np.any((occ > 1e-8) & (occ < 2 - 1e-8))):
        raise CheckpointError(
            f"{path} holds an open-shell restricted wavefunction; minpop-rks "
            f"analyses closed shells only (use minpop-rohf for ROHF orbitals)")
    if np.any((occ > 1e-8) & (np.abs(occ - 1) > 1e-8) & (np.abs(occ - 2) > 1e-8)):
        raise CheckpointError(
            f"{path}: fractional orbital occupations; MinPop needs an "
            f"integer-occupation SCF solution")


def _check_explicit(path, mol, meta, explicit):
    """Explicitly given SCF-defining flags must agree with the checkpoint."""
    bad = []
    ex = explicit or {}
    if "charge" in ex and int(ex["charge"]) != int(mol.charge):
        bad.append(f"-charge {ex['charge']} (checkpoint: {mol.charge})")
    if "mult" in ex and int(ex["mult"]) != mol.spin + 1:
        bad.append(f"-mult {ex['mult']} (checkpoint: {mol.spin + 1})")
    if "cartesian" in ex and bool(ex["cartesian"]) != bool(mol.cart):
        bad.append(f"{'-cartesian' if ex['cartesian'] else '-spherical'} "
                   f"(checkpoint: {'Cartesian' if mol.cart else 'spherical'})")
    if meta:
        for key, flag in (("basis", "-basis"), ("basis_light", "-basis-light"),
                          ("xc", "-xc")):
            if key in ex and ex[key] is not None and meta.get(key) not in (None, "") \
                    and _norm(ex[key]) != _norm(meta[key]):
                bad.append(f"{flag} {ex[key]} (checkpoint: {meta[key]})")
        if ex.get("basis_light") and not meta.get("basis_light"):
            bad.append(f"-basis-light {ex['basis_light']} (checkpoint: none)")
        want_x2c = bool(ex.get("x2c") or ex.get("x2c_atomic"))
        if ("x2c" in ex or "x2c_atomic" in ex) and want_x2c != bool(meta.get("x2c")):
            bad.append(f"-x2c (checkpoint: "
                       f"{'X2C' if meta.get('x2c') else 'non-relativistic'})")
        if ex.get("x2c_atomic") and meta.get("x2c") and not meta.get("x2c_atomic"):
            bad.append("-x2c-atomic (checkpoint: molecular X2C)")
        g = ex.get("grid_level")
        if g is not None and meta.get("grid_level", -1) not in (-1, int(g)):
            bad.append(f"-grid-level {g} (checkpoint: {meta['grid_level']})")
        if ex.get("ecp") and _norm(ex["ecp"]) not in _norm(meta.get("ecp", "")):
            bad.append(f"-ecp {ex['ecp']} (checkpoint: {meta.get('ecp') or 'none'})")
    if bad:
        raise CheckpointError(
            f"{path} does not match the command line: " + "; ".join(bad)
            + ". With -read-chk these settings come from the checkpoint. "
              "Drop the flag, or rerun the SCF without -read-chk.")


def check_geometry(path, mol, atom_str, tol=1e-3):
    """An -xyz given together with -read-chk must be the same molecule.
    Compared through the interatomic distances, so the frame does not matter."""
    lines = [l.split() for l in str(atom_str).replace(";", "\n").splitlines()
             if l.strip()]
    sym = [l[0] for l in lines]
    xyz = np.array([[float(x) for x in l[1:4]] for l in lines])
    mine = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    norm = lambda s: re.sub(r"[^A-Za-z]", "", s).capitalize()
    if [norm(s) for s in sym] != [norm(s) for s in mine]:
        raise CheckpointError(
            f"{path} holds {' '.join(mine)} but the -xyz file lists "
            f"{' '.join(sym)} (same atoms in the same order are required)")
    ref = mol.atom_coords(unit="Angstrom")
    d1 = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    d2 = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    err = float(abs(d1 - d2).max()) if len(sym) > 1 else 0.0
    if err > tol:
        raise CheckpointError(
            f"{path} was computed at a different geometry than the -xyz file "
            f"(largest interatomic-distance difference {err:.4f} A)")


def restore(path, method, verbose=True, explicit=None, labels=None,
            atom_str=None):
    """Rebuild the mean-field object the analysis needs, without an SCF.

    Returns (mf, meta). Prints the SCF section of the report: the stored one
    when the checkpoint has it, otherwise one rebuilt from the molecule.
    """
    from pyscf import scf, dft
    if method not in METHODS:
        raise CheckpointError(f"unknown method {method!r}")
    mol, rec, meta = load(path)
    have = stored_method(rec, meta)
    if have and have != method:
        raise CheckpointError(
            f"{path} holds a {'GVB-PP' if have == 'GVBPP' else have} wavefunction"
            + (f" (xc={meta.get('xc')})" if meta.get("xc") else "")
            + f", not {method}. Use {METHODS[have][3]} -read-chk {path}, or "
              f"`minpop-chk {path}`, which picks the driver itself.")
    _check_wavefunction(path, mol, rec, method)
    _check_explicit(path, mol, meta, explicit)
    if atom_str is not None:
        check_geometry(path, mol, atom_str)

    cls = {"ROHF": scf.ROHF, "UHF": scf.UHF, "RKS": dft.RKS, "UKS": dft.UKS}[method]
    mf = cls(mol)
    if METHODS[method][1]:
        xc = (meta or {}).get("xc") or (labels or {}).get("xc")
        if xc:
            mf.xc = xc
    mf.mo_coeff = np.asarray(rec["mo_coeff"])
    mf.mo_occ = np.asarray(rec["mo_occ"])
    mf.mo_energy = np.asarray(rec["mo_energy"])
    mf.e_tot = float(rec["e_tot"])
    mf.converged = bool((meta or {}).get("converged", True))

    if verbose:
        header = (meta or {}).get("header") or ""
        how = "the SCF section below is the stored one"
        if "Molecule:" not in header or "converged SCF energy" not in header:
            header = _synthetic_header(path, mol, rec, meta, method, labels)
            how = "the SCF section below is rebuilt from the file"
        print(f"Checkpoint file: {path} (read; SCF not rerun, {how})")
        sys.stdout.write(header if header.endswith("\n") else header + "\n")
    return mf, meta


def load_gvbpp(path):
    """The GVB-PP record of a checkpoint as a dict, or None when the file
    holds none."""
    from pyscf import lib
    try:
        rec = lib.chkfile.load(path, GVB_GROUP)
    except Exception:
        rec = None
    if not rec or "mo_coeff" not in rec:
        return None
    return {k: (np.asarray(v) if k in ("mo_coeff", "occupations", "overlaps")
                else _plain(v)) for k, v in rec.items()}


def restore_gvbpp(path, verbose=True, explicit=None, atom_str=None,
                  n_pairs=None):
    """Rebuild what minpop_gvbpp() needs from a GVB-PP checkpoint, without an
    SCF or a GVB-PP optimization. Returns (g, meta); g carries mol, ncore,
    n_pairs, n_open, mo_coeff, occupations, overlaps, e_tot, e_corr and
    converged, like a gvbpp.GVBPP object after kernel()."""
    import types
    mol, rec, meta = load(path)
    gvb = load_gvbpp(path)
    have = stored_method(rec, meta)
    if gvb is None:
        raise CheckpointError(
            f"{path} holds no GVB-PP wavefunction"
            + (f" (it is a {have} checkpoint: use {METHODS[have][3]} -read-chk "
               f"{path})" if have in METHODS else
               " (a plain PySCF chkfile has only SCF orbitals)")
            + ". To run GVB-PP starting from it, use -guess-chk.")
    if n_pairs is not None and int(n_pairs) != int(gvb["n_pairs"]):
        raise CheckpointError(
            f"{path} holds a GVB-PP wavefunction with {gvb['n_pairs']} pairs, "
            f"not -npairs {n_pairs}. With -read-chk the pair count comes from "
            f"the checkpoint. Drop the flag, or rerun without -read-chk.")
    nact = int(gvb["ncore"]) + 2 * int(gvb["n_pairs"]) + int(gvb["n_open"])
    c = gvb["mo_coeff"]
    if c.shape[0] != mol.nao_nr() or c.shape[1] < nact:
        raise CheckpointError(f"{path}: GVB-PP orbitals do not fit the stored "
                              f"molecule ({c.shape} for {mol.nao_nr()} functions)")
    _check_explicit(path, mol, meta, explicit)
    if atom_str is not None:
        check_geometry(path, mol, atom_str)
    g = types.SimpleNamespace(
        mol=mol, mo_coeff=c, occupations=gvb["occupations"],
        overlaps=gvb["overlaps"], ncore=int(gvb["ncore"]),
        n_pairs=int(gvb["n_pairs"]), n_open=int(gvb["n_open"]),
        e_tot=float(gvb["e_tot"]), e_corr=float(gvb["e_corr"]),
        converged=bool(gvb["converged"]))
    if verbose:
        header = (meta or {}).get("header") or ""
        how = "the SCF and GVB-PP section below is the stored one"
        if "Molecule:" not in header or "converged GVB-PP energy" not in header:
            m = meta or {}
            basis = m.get("basis") or (mol.basis if isinstance(mol.basis, str)
                                       else "custom")
            if m.get("basis_light"):
                z = m.get("light_max_z", 18)
                basis = f"{basis} on Z>{z}, {m['basis_light']} on Z<={z}"
            out = [f"Molecule: {m.get('xyz_file') or path}",
                   f"Charge: {mol.charge}, Multiplicity: {mol.spin + 1}",
                   f"Basis: {basis} (5D/7F spherical)",
                   f"Atoms: {mol.natm}, Electrons: {mol.nelectron}", ""]
            if m.get("x2c"):
                out.append("Hamiltonian: spin-free X2C-1e (scalar relativistic)"
                           + (", atomic (atom-by-atom) decoupling"
                              if m.get("x2c_atomic") else ""))
            out += [_energy_line(float(rec["e_tot"])),
                    f"GVB-PP: {g.n_pairs} pairs",
                    f"converged GVB-PP energy = {g.e_tot:.10f}", ""]
            header = "\n".join(out) + "\n"
            how = "the SCF and GVB-PP section below is rebuilt from the file"
        print(f"Checkpoint file: {path} (read; SCF and GVB-PP not rerun, {how})")
        sys.stdout.write(header if header.endswith("\n") else header + "\n")
    return g, meta


# --------------------------------------------------------------------------- #
# A checkpoint as the initial guess of a new SCF (-guess-chk)
# --------------------------------------------------------------------------- #
def _align(old, new):
    """Rigid motion that carries the checkpoint geometry onto the current one.
    Returns (R, rmsd, moved) with moved = (old - centroid) @ R.T + centroid',
    all in Bohr. R is a proper rotation (Kabsch)."""
    co, cn = old.mean(axis=0), new.mean(axis=0)
    P, Q = old - co, new - cn
    if len(old) == 1:
        R = np.eye(3)
    else:
        U, _, Vt = np.linalg.svd(P.T @ Q)
        d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    moved = P @ R.T + cn
    rmsd = float(np.sqrt(((moved - new) ** 2).sum(axis=1).mean()))
    return R, rmsd, moved


def _cart_rotation(l, R):
    """Rotation of one Cartesian shell (PySCF order: xx, xy, xz, yy, yz, zz
    for d). Column k holds the rotated function k in the unrotated ones; the
    monomials x^a y^b z^c mix among themselves under a rotation, and the
    matrix follows from their values at a set of points."""
    pw = [(a, b, l - a - b) for a in range(l, -1, -1) for b in range(l - a, -1, -1)]
    rng = np.random.default_rng(7)
    r = rng.standard_normal((4 * len(pw) + 8, 3))
    val = lambda x: np.array([x[:, 0] ** a * x[:, 1] ** b * x[:, 2] ** c
                              for a, b, c in pw]).T
    M, *_ = np.linalg.lstsq(val(r), val(r @ R), rcond=None)
    return M


def _ao_rotation(mol, R):
    """T with rotated coefficients = T @ coefficients, for atoms moved by
    new = old @ R.T. Checked against the SCF energy of a rotated molecule."""
    import scipy.linalg as sl
    if not mol.cart:
        if not hasattr(mol, "ao_rotation_matrix"):
            raise CheckpointError(
                "this PySCF cannot rotate orbitals (Mole.ao_rotation_matrix is "
                "missing). Run with -no-std-orient and an -xyz in the "
                "checkpoint's own frame, or update PySCF.")
        return np.asarray(mol.ao_rotation_matrix(R)).T
    blocks, cache = [], {}
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        if l not in cache:
            cache[l] = _cart_rotation(l, R)
        blocks.extend([cache[l]] * mol.bas_nctr(i))
    return sl.block_diag(*blocks)


def _check_same_system(path, cmol, mol):
    norm = lambda sym: re.sub(r"[^A-Za-z]", "", sym).capitalize()
    mine = [norm(mol.atom_pure_symbol(i)) for i in range(mol.natm)]
    theirs = [norm(cmol.atom_pure_symbol(i)) for i in range(cmol.natm)]
    if mine != theirs:
        raise CheckpointError(
            f"-guess-chk {path} holds {' '.join(theirs)} but this run is "
            f"{' '.join(mine)} (same atoms in the same order are required)")
    if cmol.charge != mol.charge or cmol.spin != mol.spin:
        raise CheckpointError(
            f"-guess-chk {path} has charge {cmol.charge}, multiplicity "
            f"{cmol.spin + 1}; this run has charge {mol.charge}, multiplicity "
            f"{mol.spin + 1}")


def _transfer(cmol, mol, orbs):
    """Carry orbital sets from the checkpoint molecule to the current one:
    rotate them when the geometries differ by a rigid motion, project them
    when the basis or the geometry differs. Not orthonormalized. Returns
    (orbital sets, info) with info = {rotated, projected, rmsd (Bohr)}."""
    from pyscf import gto
    from pyscf.scf import addons
    R, rmsd, moved = _align(cmol.atom_coords(), mol.atom_coords())
    rotated = abs(R - np.eye(3)).max() > 1e-8
    if rotated:
        T = _ao_rotation(cmol, R)
        orbs = [T @ o for o in orbs]
    if abs(moved - cmol.atom_coords()).max() > 1e-10:
        cmol = cmol.copy()
        cmol.set_geom_(moved, unit="Bohr")
    same_basis = (bool(cmol.cart) == bool(mol.cart)
                  and gto.same_basis_set(cmol, mol)
                  and cmol.nao_nr() == mol.nao_nr())
    project = not (same_basis and rmsd < 1e-6)
    if project:
        orbs = [addons.project_mo_nr2nr(cmol, o, mol) if o.shape[1] else
                np.zeros((mol.nao_nr(), 0)) for o in orbs]
    return orbs, {"rotated": rotated, "projected": project, "rmsd": rmsd}


def _lowdin(o, S, path):
    """Symmetric orthonormalization: the closest orthonormal set, same order."""
    w, v = np.linalg.eigh(o.T @ S @ o)
    if w.min() < 1e-6:
        raise CheckpointError(
            f"-guess-chk {path}: the stored orbitals become linearly "
            f"dependent in the current basis, so they are no use as a guess")
    return o @ (v / np.sqrt(w)) @ v.T


def _how(info):
    steps = []
    if info["rotated"]:
        steps.append("rotated into the current frame")
    if info["projected"]:
        steps.append("projected onto the current basis" if info["rmsd"] < 1e-6
                     else "projected onto the current basis and geometry")
    return ", ".join(steps) if steps else "used as stored"


def _source_label(cmol, rec, meta):
    m = meta or {}
    c = np.asarray(rec["mo_coeff"])
    src = m.get("method") or ("unrestricted" if c.ndim == 3 else "restricted")
    if src == "GVBPP":
        src = "RHF/ROHF reference of a GVB-PP run"
    if m.get("xc"):
        src += f"/{m['xc']}"
    basis = m.get("basis") or (cmol.basis if isinstance(cmol.basis, str)
                               else "custom basis")
    if m.get("basis_light"):
        basis += f" + {m['basis_light']}"
    return src, basis


def guess_density(path, mol, method, verbose=True):
    """Initial-guess density for a new SCF on `mol` from the wavefunction in
    the checkpoint `path` (a MinPop checkpoint or a plain PySCF chkfile).

    The stored occupied orbitals are rotated into the current frame when the
    two geometries differ by a rigid motion, projected onto the current basis
    when the basis differs, and orthonormalized, so the guess always holds the
    right number of electrons. Method and functional may differ from the
    current run; atoms, charge and multiplicity may not. For a GVB-PP
    checkpoint the orbitals are those of its RHF/ROHF reference.

    Returns the density in the form the driver's SCF takes: (alpha, beta) for
    ROHF, UHF and UKS, the total density for RKS (and for a closed-shell RHF
    reference, method='RHF').
    """
    if method not in METHODS and method != "RHF":
        raise CheckpointError(f"unknown method {method!r}")
    cmol, rec, meta = load(path)
    _check_same_system(path, cmol, mol)
    c = np.asarray(rec["mo_coeff"])
    occ = np.asarray(rec["mo_occ"])
    if c.ndim == 3:
        orbs = [c[0][:, occ[0] > 0.5], c[1][:, occ[1] > 0.5]]
    else:
        orbs = [c[:, occ > 0.5], c[:, occ > 1.5]]
    orbs, info = _transfer(cmol, mol, orbs)
    S = mol.intor("int1e_ovlp")
    dms = []
    for o in orbs:
        if o.shape[1] == 0:
            dms.append(np.zeros_like(S))
            continue
        o = _lowdin(o, S, path)
        dms.append(o @ o.T)
    dma, dmb = dms
    if verbose:
        src, basis = _source_label(cmol, rec, meta)
        print(f"Initial guess: checkpoint file {path} ({src}, {basis}, "
              f"E = {float(rec['e_tot']):.9f}; {_how(info)})")
        if info["rmsd"] > 1e-3:
            print(f"Initial guess: the checkpoint geometry differs from this "
                  f"one, RMSD {info['rmsd'] * 0.529177210903:.4f} A after alignment")
    if method in ("RKS", "RHF"):
        return dma + dmb
    return np.array([dma, dmb])


def gvb_start_orbitals(path, mol, mf, n_pairs, verbose=True):
    """Starting orbitals for a GVB-PP optimization on `mol` from the GVB-PP
    wavefunction stored in `path`, or None when the file has none that fits
    (no GVB-PP record, another pair count, another core).

    The stored core, pair (bonding, correlating, ...) and open-shell natural
    orbitals are carried over like an SCF guess (rotated, projected), then
    orthonormalized in their stored order, which is the order gvbpp expects
    of a user start. The virtual space is the orthogonal complement, built
    from the reference orbitals in `mf`.
    """
    gvb = load_gvbpp(path)
    say = print if verbose else (lambda *a, **k: None)
    if gvb is None:
        return None
    cmol, rec, meta = load(path)
    _check_same_system(path, cmol, mol)
    occ = np.asarray(mf.mo_occ)
    ndocc, nopen = int((occ > 1.5).sum()), int(((occ > 0.5) & (occ < 1.5)).sum())
    ncore = ndocc - int(n_pairs)
    if int(gvb["n_pairs"]) != int(n_pairs):
        say(f"GVB-PP start: {path} holds {gvb['n_pairs']} pairs, this run has "
            f"{n_pairs}; its GVB-PP orbitals are not used (the reference SCF "
            f"still starts from it)")
        return None
    if int(gvb["ncore"]) != ncore or int(gvb["n_open"]) != nopen:
        say(f"GVB-PP start: {path} has {gvb['ncore']} core orbitals, this run "
            f"has {ncore} (different ECP or core); its GVB-PP orbitals are "
            f"not used")
        return None
    nact = ncore + 2 * int(n_pairs) + nopen
    (o,), info = _transfer(cmol, mol, [gvb["mo_coeff"][:, :nact]])
    S = mol.intor("int1e_ovlp")
    o = _lowdin(o, S, path)
    # virtual space: what is left of the reference orbitals
    cref = np.asarray(mf.mo_coeff)
    rest = cref - o @ (o.T @ S @ cref)
    w, v = np.linalg.eigh(rest.T @ S @ rest)
    nvir = cref.shape[1] - nact
    keep = np.argsort(w)[::-1][:nvir]
    if nvir and w[keep].min() < 1e-8:
        raise CheckpointError(f"-guess-chk {path}: cannot complete the "
                              f"virtual space around the stored GVB-PP orbitals")
    vir = rest @ (v[:, keep] / np.sqrt(w[keep]))
    say(f"GVB-PP start: natural orbitals of {path} ({gvb['n_pairs']} pairs, "
        f"E = {float(gvb['e_tot']):.9f}; {_how(info)})")
    return np.hstack([o, vir])


# --------------------------------------------------------------------------- #
# What the drivers call
# --------------------------------------------------------------------------- #
def run_with_checkpoint(method, scf_func, analyse_func, scf_kwargs,
                        analyse_kwargs, chk=None, read_chk=None, explicit=None,
                        read_xyz=None):
    """Get the wavefunction (SCF or checkpoint), then run the MinPop analysis.

    method         : 'ROHF', 'UHF', 'RKS', 'UKS' or 'GVBPP'
    scf_func       : the driver's SCF routine; returns the converged mf
                     (for 'GVBPP': the converged GVBPP object, whose .mf is
                     the reference)
    analyse_func   : the driver's minpop_<method>(mf, ...)
    scf_kwargs     : keyword arguments of scf_func
    analyse_kwargs : keyword arguments of analyse_func
    chk            : write the converged SCF here (None: no file)
    read_chk       : read the SCF from here and skip it
    explicit       : SCF-defining flags given on the command line (CLI only)
    read_xyz       : the driver's XYZ reader, to compare -xyz with the checkpoint
    """
    verbose = scf_kwargs.get("verbose", True)
    if read_chk:
        atom_str = None
        if scf_kwargs.get("xyz_file") and read_xyz is not None:
            atom_str = read_xyz(scf_kwargs["xyz_file"])
        if method == "GVBPP":
            g, _ = restore_gvbpp(read_chk, verbose=verbose, explicit=explicit,
                                 atom_str=atom_str,
                                 n_pairs=(explicit or {}).get("n_pairs"))
            return analyse_func(g, **analyse_kwargs)
        mf, _ = restore(read_chk, method, verbose=verbose, explicit=explicit,
                        labels=explicit, atom_str=atom_str)
        return analyse_func(mf, **analyse_kwargs)

    if chk:
        with recording() as buf:
            mf = scf_func(**scf_kwargs)
        try:
            if method == "GVBPP":      # scf_func returned the GVBPP object
                save(mf.mf, chk, method, header=buf.getvalue(),
                     settings=scf_kwargs, gvb=mf)
            else:
                save(mf, chk, method, header=buf.getvalue(), settings=scf_kwargs)
            if verbose:
                print(f"Checkpoint file: {chk} (written; reuse with -read-chk)")
                print()
        except Exception as exc:          # never lose a finished SCF over I/O
            print(f"[chk] could not write {chk}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    else:
        mf = scf_func(**scf_kwargs)
    return analyse_func(mf, **analyse_kwargs)


def add_arguments(parser):
    """The checkpoint flags every driver shares."""
    g = parser.add_argument_group("checkpoint")
    g.add_argument("-chk", dest="chk", metavar="PATH", default=None,
                   help="Write the converged SCF (molecule, basis, orbitals, "
                        "energy, settings) to this PySCF checkpoint file. "
                        "Default: <xyz name>_<method>.chk in the current "
                        "directory, e.g. rhco3_uks.chk. Give a name whenever "
                        "runs of the same molecule and method share a "
                        "directory, so they do not overwrite each other.")
    g.add_argument("-no-chk", dest="no_chk", action="store_true", default=False,
                   help="Do not write a checkpoint file.")
    g.add_argument("-read-chk", dest="read_chk", metavar="PATH", default=None,
                   help="Skip the SCF: read the wavefunction from this "
                        "checkpoint and project it. -xyz is then optional. "
                        "Method, basis, charge, multiplicity, functional and "
                        "Hamiltonian come from the file; a flag for one of "
                        "them that disagrees with the file stops the run. "
                        "Projection and output flags (-minimal-basis, "
                        "-light-max-z, -label-style, -symmetry, "
                        "-symmetry-shells, -no-azimuthal-gauge, -json, "
                        "-csv-long) apply as usual. A plain PySCF chkfile "
                        "works too; -basis, -xc and -x2c then only label "
                        "the report.")
    g.add_argument("-guess-chk", dest="guess_chk", metavar="PATH", default=None,
                   help="Run the SCF, starting it from the wavefunction in "
                        "this checkpoint instead of an atomic guess. The "
                        "file may come from another basis, method or "
                        "functional (HF orbitals for a DFT run, a small "
                        "basis for a large one, a restricted solution for "
                        "an unrestricted run) and from a rotated or "
                        "slightly different geometry: the orbitals are "
                        "rotated, projected and orthonormalized as needed. "
                        "Atoms, charge and multiplicity must match. The "
                        "SCF, -stable and the analysis run as usual and a "
                        "new checkpoint is written. Replaces -guess, "
                        "-guessmix and -guess-triplet. Not the same as "
                        "-read-chk, which runs no SCF.")
    return g


def resolve_cli(parser, args, method, argv=None):
    """Validate the checkpoint flags of a driver. Returns (chk, read_chk,
    explicit, xyz): xyz is the geometry file to hand to the run (None when a
    reprojection was given no -xyz). Fills args.xyz_file from the checkpoint
    when -xyz is absent, for the exports."""
    explicit = explicit_args(parser, argv)
    if args.read_chk:
        if args.chk:
            parser.error("-chk and -read-chk together: a reprojection does "
                         "not rerun the SCF, so there is nothing new to write")
        ignored = sorted(SCF_PROCEDURE[k] for k in explicit if k in SCF_PROCEDURE)
        if ignored:
            print(f"[chk] no SCF is run with -read-chk; ignored: "
                  f"{', '.join(ignored)}", file=sys.stderr)
        if not args.xyz_file:
            try:
                _, _, meta = load(args.read_chk)
            except CheckpointError as exc:
                sys.exit(f"MinPop ERROR: {exc}")
            args.xyz_file = (meta or {}).get("xyz_file") or args.read_chk
            scf_xyz = None
        else:
            scf_xyz = args.xyz_file
        return None, args.read_chk, explicit, scf_xyz
    if not args.xyz_file:
        parser.error("give -xyz FILE (run the SCF) or -read-chk FILE "
                     "(reuse a stored one)")
    if args.guess_chk:
        clash = [f for k, f in (("guess", "-guess"), ("guessmix", "-guessmix"),
                                ("guess_triplet", "-guess-triplet"))
                 if k in explicit]
        if clash:
            parser.error(f"-guess-chk already sets the starting point of the "
                         f"SCF; drop {', '.join(clash)}")
        if not os.path.isfile(args.guess_chk):
            parser.error(f"-guess-chk: file not found: {args.guess_chk}")
    chk = None if args.no_chk else (args.chk or default_path(args.xyz_file, method))
    return chk, None, explicit, args.xyz_file


# --------------------------------------------------------------------------- #
# minpop-chk
# --------------------------------------------------------------------------- #
def describe(path):
    """Text summary of a checkpoint."""
    mol, rec, meta = load(path)
    m = meta or {}
    unres = _unrestricted(rec)
    out = [f"Checkpoint file: {path}"]
    if meta:
        out.append(f"  written by MinPop       : yes (minpop-chk format "
                   f"{m.get('format_version')}, PySCF {m.get('pyscf_version')}, "
                   f"{m.get('created_utc')})")
        out.append(f"  method                  : "
                   f"{'GVB-PP' if m.get('method') == 'GVBPP' else m.get('method')}"
                   + (f"  xc={m['xc']}  grid level="
                      f"{'default' if m.get('grid_level', -1) == -1 else m['grid_level']}"
                      if m.get("xc") else ""))
        basis = m.get("basis")
        if m.get("basis_light"):
            basis = (f"{basis} on Z>{m.get('light_max_z')}, "
                     f"{m['basis_light']} on Z<={m.get('light_max_z')}")
        out.append(f"  computational basis     : {basis} "
                   f"({'Cartesian' if mol.cart else 'spherical'})")
        out.append("  Hamiltonian             : "
                   + ("sfX2C-1e" + (", atomic decoupling" if m.get("x2c_atomic") else "")
                      if m.get("x2c") else "non-relativistic"))
        out.append(f"  ECP                     : {m.get('ecp') or 'none'}")
        out.append(f"  geometry file           : {m.get('xyz_file') or 'unknown'}"
                   f"  (stored frame: "
                   f"{'Gaussian standard orientation' if m.get('standard_orientation') else 'as given'})")
        if m.get("guess_chk"):
            out.append(f"  SCF started from        : {m['guess_chk']}")
        out.append(f"  original command        : {m.get('command_line')}")
    else:
        out.append("  written by MinPop       : no (plain PySCF chkfile; "
                   "method, functional and Hamiltonian are not recorded)")
        out.append(f"  orbitals                : "
                   f"{'unrestricted' if unres else 'restricted'}")
        out.append(f"  computational basis     : "
                   f"{mol.basis if isinstance(mol.basis, str) else 'custom'} "
                   f"({'Cartesian' if mol.cart else 'spherical'})")
    out.append(f"  charge, multiplicity    : {mol.charge}, {mol.spin + 1}")
    out.append(f"  atoms, electrons        : {mol.natm}, {mol.nelectron}  "
               f"({' '.join(mol.atom_pure_symbol(i) for i in range(mol.natm))})")
    out.append(f"  basis functions         : {mol.nao_nr()}")
    out.append(f"  SCF energy (Eh)         : {float(rec['e_tot']):.12f}")
    if unres:
        ss, s = _spin_square(mol, rec["mo_coeff"], rec["mo_occ"])
        out.append(f"  <S^2>, 2S+1             : {ss:.6f}, {s:.6f}")
    gvb = load_gvbpp(path)
    if gvb is not None:
        out[-1] = out[-1].replace("SCF energy (Eh)         ", "reference SCF energy    ")
        out.append(f"  GVB-PP energy (Eh)      : {float(gvb['e_tot']):.12f}  "
                   f"(correlation {float(gvb['e_corr']):.6f})")
        out.append(f"  GVB-PP space            : {gvb['ncore']} core, "
                   f"{gvb['n_pairs']} pairs, {gvb['n_open']} open-shell orbitals")
        o = np.asarray(gvb["occupations"]).reshape(-1, 2)
        out.append("  pair occupations        : "
                   + "  ".join(f"{a:.4f}/{b:.4f}" for a, b in o))
    if meta:
        out.append(f"  reproject with          : {METHODS[m['method']][3]} "
                   f"-read-chk {path} -minimal-basis NAME   (or minpop-chk "
                   f"{path} -minimal-basis NAME)")
    return "\n".join(out)


def main(argv=None):
    """minpop-chk: reproject a checkpoint, or show what it holds."""
    parser = argparse.ArgumentParser(
        prog="minpop-chk",
        description="Reproject a MinPop / PySCF checkpoint onto a minimal "
                    "basis without rerunning the SCF. The driver that wrote "
                    "the checkpoint (ROHF, UHF, RKS, UKS or GVB-PP) is chosen "
                    "from "
                    "the file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Every other flag goes to the driver unchanged. The ones that matter here:
  -minimal-basis NAME   sto-3g (default), sto-3g-ano, sto-3g-ano-occ, ano-rcc-mb,
                        ano-rcc-mb-occ, minao, or LIGHT+HEAVY
  -light-max-z Z        highest Z on the LIGHT minimal basis (default 18)
  -label-style STYLE    gaussian | physical
  -symmetry             sigma/pi/delta/phi bond populations, condensed to atoms
  -symmetry-shells      the same, also condensed to shells
  -no-azimuthal-gauge   keep the stored azimuth of a linear molecule
  -xyz FILE             optional: checked against the stored geometry
  -json PATH, -json-xyz-dir DIR, -csv-long PATH, -csv-sparse-tol X
  -q                    suppress the report
For a plain PySCF chkfile also: -xc NAME, -x2c, -basis NAME (labels only).

Examples:
  minpop-chk rhco3_uks.chk -info
  minpop-chk rhco3_uks.chk -minimal-basis sto-3g-ano-occ -symmetry-shells
  minpop-chk rhco3_uks.chk -minimal-basis minao -json rhco3_minao.json
  minpop-chk pyscf_run.chk -method uks -xc b3lyp -minimal-basis sto-3g
""")
    parser.add_argument("chk", metavar="FILE", help="checkpoint file")
    parser.add_argument("-info", action="store_true",
                        help="print what the checkpoint holds and exit")
    parser.add_argument("-method", choices=["rohf", "uhf", "rks", "uks"],
                        default=None,
                        help="driver for a plain PySCF chkfile (default: "
                             "uhf for unrestricted orbitals, rohf for "
                             "restricted; uks/rks when -xc is given). A MinPop "
                             "checkpoint names its own method.")
    args, rest = parser.parse_known_args(argv)
    try:
        if args.info:
            print(describe(args.chk))
            return 0
        _, rec, meta = load(args.chk)
    except CheckpointError as exc:
        sys.exit(f"MinPop ERROR: {exc}")
    method = stored_method(rec, meta)
    if method and args.method and args.method.upper() != method:
        sys.exit(f"MinPop ERROR: {args.chk} holds a {method} wavefunction; "
                 f"-method {args.method} does not apply")
    if not method:
        if args.method:
            method = args.method.upper()
        else:
            ks = "-xc" in rest
            method = ("UKS" if ks else "UHF") if _unrestricted(rec) else \
                     ("RKS" if ks else "ROHF")
    import importlib
    modname = METHODS[method][2]
    try:
        driver = importlib.import_module(f"minpop.{modname}")
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        driver = importlib.import_module(modname)
    # the driver echoes sys.argv as its command line: make that a command the
    # user can run again as it stands
    sys.argv = [driver.__file__, "-read-chk", args.chk] + rest
    return driver.main()


if __name__ == "__main__":
    sys.exit(main())
