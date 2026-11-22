"""Fast regression tests (about a minute on one core). Run: pytest -q"""
import os
import subprocess
import sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
XYZ = os.path.join(HERE, "..", "examples", "xyz")
PKG = os.path.join(HERE, "..", "minpop")
O_CHARGE_H2O_631G = -0.611678   # ROHF/6-31G -> STO-3G MBS, Gaussian-validated path


import tempfile
WORK = tempfile.mkdtemp(prefix="minpop_tests_")   # default checkpoints land here


def run(script, *args, ok=True):
    cmd = [sys.executable, os.path.join(PKG, script), *args]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                         cwd=WORK, env=env)
    if not ok:
        assert out.returncode != 0
        return out.stderr
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out.stdout


def charges(text):
    lines = text.splitlines()
    i = [k for k, l in enumerate(lines) if "MBS Mulliken charges and spin" in l][0]
    rows = []
    for l in lines[i + 2:]:
        if not l.strip() or l.startswith(" Sum"):
            break
        rows.append([float(x) for x in l.split()[2:4]])
    return np.array(rows)


def test_rohf_h2o_sto3g_reference():
    """Gaussian-MBS convention path: values fixed by the validated STO-3G code."""
    q = charges(run("minpop_rohf.py", "-xyz", f"{XYZ}/h2o.xyz", "-basis", "6-31G"))
    assert abs(q[0, 0] - O_CHARGE_H2O_631G) < 1e-4   # value from the validated code
    assert abs(q.sum(0)[0]) < 1e-5                     # charges sum to zero


def test_uhf_ch2_triplet_spin():
    q = charges(run("minpop_uhf.py", "-xyz", f"{XYZ}/ch2.xyz", "-mult", "3",
                    "-basis", "6-31G"))
    assert abs(q[:, 1].sum() - 2.0) < 1e-5


def test_anorcc_truncations_shapes():
    sys.path.insert(0, os.path.join(PKG, "basis_sets"))
    from anorcc_basis_pyscf import ANO_RCC_MB, ANO_RCC_VDZP
    assert [(s[0], len(s[1]) - 1) for s in ANO_RCC_MB["Pt"]] == [(0, 6), (1, 5), (2, 3), (3, 1)]
    assert [(s[0], len(s[1]) - 1) for s in ANO_RCC_VDZP["H"]] == [(0, 2), (1, 1)]
    assert [(s[0], len(s[1]) - 1) for s in ANO_RCC_MB["Fe"]] == [(0, 4), (1, 3), (2, 1)]


def test_ano_rcc_mb_projection_h2o():
    out = run("minpop_rohf.py", "-xyz", f"{XYZ}/h2o.xyz", "-basis", "ano-rcc-vdzp",
              "-minimal-basis", "ano-rcc-mb")
    assert "ANO-RCC-MB" in out
    q = charges(out)
    assert abs(q.sum(0)[0]) < 1e-5


def test_bond_symmetry_selftest():
    sys.path.insert(0, PKG)
    import bond_symmetry
    from pyscf import gto, scf
    mol = gto.M(atom="C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238; "
                     "H 0 0.923 -1.238; H 0 -0.923 -1.238", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    comps = bond_symmetry.bond_components(mol, mf.make_rdm1())
    r = comps[(0, 1)]
    assert abs(r["sigma"] + r["pi"] + r["delta"] - r["total"]) < 1e-12
    assert r["sigma"] > 0.35 and r["pi"] > 0.15
    cm = bond_symmetry.class_matrices(mol, mf.make_rdm1())
    off = ~np.eye(mol.natm, dtype=bool)
    tot = sum(cm["atom"][c] for c in cm["classes"])
    assert np.abs((tot - cm["atom_total"])[off]).max() < 1e-12


def test_bond_symmetry_section_in_report():
    out = run("minpop_uhf.py", "-xyz", f"{XYZ}/ch2.xyz", "-mult", "3", "-basis", "6-31G",
              "-symmetry")
    assert "MBS σ Condensed to atoms (all electrons)" in out
    assert "MBS π Atomic-Atomic Spin Densities" in out


@pytest.mark.skipif(__import__("importlib").util.find_spec("gvbpp") is None,
                    reason="gvbpp not installed")
def test_gvbpp_h2():
    out = run("minpop_gvbpp.py", "-xyz", f"{XYZ}/h2.xyz", "-npairs", "1", "-basis", "cc-pvdz")
    assert "GVB-PP pairs:" in out
    assert "2 of 2 natural orbitals projected" in out


def test_sto3g_ano_beyond_xe():
    """sto-3g-ano must build ANO-RCC-MB on Pt and STO-3G on O without asking STO-3G for Pt."""
    out = run("minpop_rohf.py", "-xyz", f"{XYZ}/pth.xyz", "-mult", "2", "-basis", "ano-rcc-vdzp",
              "-basis-light", "6-31G*", "-x2c", "-minimal-basis", "sto-3g-ano")
    assert "STO-3G on Z<=18, ANO-RCC-MB on Z>18" in out
    assert abs(charges(out).sum(0)[0]) < 1e-5


def test_ano_rcc_mb_occ_shells():
    sys.path.insert(0, os.path.join(PKG, "basis_sets"))
    from anorcc_basis_pyscf import MB_COUNTS, MB_OCC_COUNTS
    assert MB_OCC_COUNTS["Rh"] == (5, 3, 2) and MB_COUNTS["Rh"] == (5, 4, 2)
    assert MB_OCC_COUNTS["Fe"] == (4, 2, 1) and MB_OCC_COUNTS["Ga"] == (4, 3, 1)
    assert MB_OCC_COUNTS["Pt"] == (6, 4, 3, 1) and MB_OCC_COUNTS["C"] == (2, 1)


# ---------------------------------------------------------------------------
# JSON schema 1.2: the record carries everything the report prints
# ---------------------------------------------------------------------------
import json
import re


def _run_json(script, tmp_path, *args):
    js = tmp_path / "rec.json"
    cmd = [sys.executable, os.path.join(PKG, script), *args, "-json", str(js)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env,
                         cwd=WORK)
    assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-1500:]
    return out.stdout, json.loads(js.read_text())


def test_json_uhf_matches_report(tmp_path):
    text, rec = _run_json("minpop_uhf.py", tmp_path, "-xyz", f"{XYZ}/ch2.xyz", "-mult", "3",
                          "-basis", "6-31G", "-symmetry-shells")
    assert rec["schema_version"] == "1.4"
    assert rec["method"]["name"] == "uhf" and rec["method"]["label"] == "UHF"
    assert rec["method"]["basis_full"] == "6-31G" and rec["method"]["spherical"] is True
    assert rec["method"]["minimal_basis"].startswith("STO-3G")
    assert rec["method"]["hamiltonian"]["x2c"] is False
    assert "-symmetry-shells" in rec["source"]["command_line"]
    # projection completeness: both spins, numbers as printed
    comp = {c["label"]: c for c in rec["projection"]["completeness"]}
    assert set(comp) == {"alpha", "beta"}
    m = re.search(r"Projection completeness \(alpha\): ([\d.]+) of 5 electrons captured, leak ([\d.]+)", text)
    assert np.isclose(comp["alpha"]["captured"], float(m.group(1)))
    assert np.isclose(comp["alpha"]["leak"], float(m.group(2)))
    # S^2 from PySCF and after annihilation
    assert abs(rec["scf"]["s2"] - 2.0) < 0.05 and rec["scf"]["n_alpha"] == 5 and rec["scf"]["n_beta"] == 3
    # symmetry: classes sum to the condensed matrices off the diagonal (print precision)
    sym = rec["symmetry"]
    off = ~np.eye(3, dtype=bool)
    for key, tot_key in (("condensed_to_atoms", "condensed_to_atoms"),
                         ("atomic_spin_densities", "atomic_spin_densities")):
        tot = np.array(rec["mbs"][tot_key]["data"])
        s = sum(np.array(sym[key][c]["data"]) for c in sym["classes"])
        assert np.abs((s - tot)[off]).max() < 5e-6
    assert sym["below_threshold"]["condensed_to_atoms"] == ["pi", "delta", "phi"]
    sh = sym["shells"]
    assert sh["labels"] == ["1S", "2S", "2P", "1S", "1S"] and sh["atoms"] == [1, 1, 1, 2, 3]
    # shell matrix condenses to the atom matrix
    S = np.array(sh["all_electrons"]["sigma"]["data"]); at = np.array(sh["atoms"])
    A = np.array(sym["condensed_to_atoms"]["sigma"]["data"])
    assert np.isclose(S[np.ix_(at == 1, at == 2)].sum(), A[0, 1], atol=5e-6)
    assert rec["gvb"] is None and isinstance(rec["warnings"], list)


def test_json_rohf_mixed_basis_x2c(tmp_path):
    text, rec = _run_json("minpop_rohf.py", tmp_path, "-xyz", f"{XYZ}/pth.xyz", "-mult", "2",
                          "-basis", "ano-rcc-vdzp", "-basis-light", "6-31G*", "-x2c-atomic",
                          "-minimal-basis", "sto-3g-ano-occ", "-symmetry")
    m = rec["method"]
    assert m["basis_heavy"] == "ANO-RCC-VDZP" and m["basis_light"] == "6-31G*" and m["light_max_z"] == 18
    assert m["hamiltonian"]["x2c"] and m["hamiltonian"]["atomic_decoupling"]
    assert "ANO-RCC-MB-OCC" in m["minimal_basis"]
    assert rec["scf"]["n_doubly"] == 39 and rec["scf"]["n_singly"] == 1
    assert rec["geometry"]["atomic_symbols"] == ["Pt", "H"]
    assert len(rec["projection"]["completeness"]) == 1
    assert rec["symmetry"]["atomic_spin_densities"] is not None       # doublet: spin present


def test_load_record_revives_nested_arrays(tmp_path):
    sys.path.insert(0, PKG)
    import minpop_json_csv as mj
    _, rec = _run_json("minpop_uhf.py", tmp_path, "-xyz", f"{XYZ}/ch2.xyz", "-mult", "3",
                       "-basis", "6-31G", "-symmetry")
    r = mj.load_record(tmp_path / "rec.json")
    assert isinstance(r["symmetry"]["condensed_to_atoms"]["sigma"]["data"], np.ndarray)
    assert isinstance(r["mbs"]["density_alpha"]["data"], np.ndarray)


@pytest.mark.skipif(__import__("importlib").util.find_spec("gvbpp") is None,
                    reason="gvbpp not installed")
def test_json_gvbpp(tmp_path):
    text, rec = _run_json("minpop_gvbpp.py", tmp_path, "-xyz", f"{XYZ}/h2o.xyz", "-npairs", "4",
                          "-basis", "cc-pvdz", "-symmetry")
    g = rec["gvb"]
    assert rec["method"]["name"] == "gvbpp" and g["n_pairs"] == 4 and g["n_core"] == 1
    assert g["n_projected"] == 7 and g["n_natural_orbitals"] == 9 and len(g["pairs"]) == 4
    kinds = sorted(p["kind"] for p in g["pairs"])
    assert kinds == ["bond", "bond", "lone pair", "lone pair"]
    bond = [p for p in g["pairs"] if p["kind"] == "bond"][0]
    assert bond["symmetry"] == "sigma" and bond["corr_no"] == "kept" and bond["mbs_bond_population"] > 0.3
    assert g["energy_hartree"] < rec["scf"]["energy_hartree"]          # correlation lowers E
    assert len(g["trials"]) >= 2 and min(t["energy_hartree"] for t in g["trials"]) == pytest.approx(g["energy_hartree"], abs=1e-8)
    assert rec["scf"]["n_alpha"] == 5 and rec["scf"]["n_beta"] == 5
    assert rec["projection"]["completeness"][0]["label"].startswith("natural orbitals")


# ---------------------------------------------------------------------------
# Checkpoint files: keep the SCF, reproject it without rerunning it
# ---------------------------------------------------------------------------
def _no_bookkeeping(text):
    return [l for l in text.splitlines()
            if l.strip() and not l.startswith(("Command line:", "Checkpoint file:"))]


def test_chk_written_by_default(tmp_path):
    run("minpop_rohf.py", "-xyz", f"{XYZ}/h2o.xyz", "-basis", "6-31G")
    assert os.path.isfile(os.path.join(WORK, "h2o_rohf.chk"))
    text = run("minpop_rohf.py", "-read-chk", "h2o_rohf.chk")
    assert abs(charges(text)[0, 0] - O_CHARGE_H2O_631G) < 5e-6


def test_chk_reprojection_equals_direct_run(tmp_path):
    chk = str(tmp_path / "ch2.chk")
    base = ("-xyz", f"{XYZ}/ch2.xyz", "-mult", "3", "-basis", "6-31G*")
    first = run("minpop_uks.py", *base, "-chk", chk)
    assert "Checkpoint file:" in first and "(written" in first
    direct = run("minpop_uks.py", *base, "-no-chk", "-minimal-basis", "minao",
                 "-symmetry-shells")
    again = run("minpop_uks.py", "-read-chk", chk, "-minimal-basis", "minao",
                "-symmetry-shells")
    assert "(read; SCF not rerun" in again
    a, b = _no_bookkeeping(direct), _no_bookkeeping(again)
    assert len(a) == len(b)
    assert np.allclose(charges(direct), charges(again), atol=1e-6)
    num = re.compile(r"-?\d+\.\d+")
    for la, lb in zip(a, b):
        if la.startswith("WARN"):
            continue
        fa, fb = [float(x) for x in num.findall(la)], [float(x) for x in num.findall(lb)]
        assert len(fa) == len(fb) and np.allclose(fa, fb, atol=2e-5), (la, lb)
    # same SCF, a different projection: the stored energy line is reused
    assert [l for l in b if "converged SCF energy" in l] == \
           [l for l in _no_bookkeeping(first) if "converged SCF energy" in l]


def test_chk_tool_picks_the_driver_and_json(tmp_path):
    chk = str(tmp_path / "h2o.chk")
    run("minpop_rks.py", "-xyz", f"{XYZ}/h2o.xyz", "-basis", "6-31G", "-xc", "b3lyp",
        "-chk", chk)
    info = run("minpop_chk.py", chk, "-info")
    assert "RKS" in info and "b3lyp" in info
    js = tmp_path / "r.json"
    text = run("minpop_chk.py", chk, "-minimal-basis", "ano-rcc-mb", "-json", str(js))
    assert "MinPop Analysis" in text and "ANO-RCC-MB" in text
    rec = json.loads(js.read_text())
    assert rec["source"]["checkpoint"] == {"path": chk, "mode": "read"}
    assert rec["method"]["name"] == "rks" and rec["method"]["xc"] == "b3lyp"
    assert rec["scf"]["energy_hartree"] is not None


def test_chk_refuses_a_mismatch(tmp_path):
    chk = str(tmp_path / "ch2.chk")
    run("minpop_uhf.py", "-xyz", f"{XYZ}/ch2.xyz", "-mult", "3", "-basis", "6-31G",
        "-chk", chk)
    assert "holds a UHF wavefunction" in run("minpop_uks.py", "-read-chk", chk, ok=False)
    err = run("minpop_uhf.py", "-read-chk", chk, "-basis", "cc-pVDZ", ok=False)
    assert "does not match the command line" in err
    err = run("minpop_uhf.py", "-read-chk", chk, "-xyz", f"{XYZ}/h2o.xyz", ok=False)
    assert "-xyz file lists" in err


def test_chk_plain_pyscf_file(tmp_path):
    from pyscf import gto, scf
    chk = str(tmp_path / "plain.chk")
    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
                basis="6-31g", verbose=0)
    mf = scf.RHF(mol)
    mf.chkfile = chk
    mf.conv_tol = 1e-10
    mf.kernel()
    text = run("minpop_chk.py", chk)
    assert "rebuilt from the file" in text
    assert abs(charges(text)[0, 0] - O_CHARGE_H2O_631G) < 2e-3


def _scf_energy(text):
    return float(re.findall(r"converged SCF energy = (-?\d+\.\d+)", text)[-1])


def test_guess_chk_carries_a_broken_symmetry_state(tmp_path):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nstretched\nH 0 0 0\nH 0 0 2.0\n")
    chk = str(tmp_path / "bs.chk")
    run("minpop_uks.py", "-xyz", str(xyz), "-basis", "6-31G*", "-guessmix", "-stable",
        "-chk", chk)
    plain = run("minpop_uks.py", "-xyz", str(xyz), "-basis", "cc-pVDZ", "-no-chk")
    seeded = run("minpop_uks.py", "-xyz", str(xyz), "-basis", "cc-pVDZ", "-no-chk",
                 "-guess-chk", chk)
    assert "Initial guess: checkpoint file" in seeded
    assert "projected onto the current basis" in seeded
    assert abs(charges(plain)[0, 1]) < 1e-6          # restricted solution
    assert abs(charges(seeded)[0, 1]) > 0.5          # broken symmetry kept
    assert _scf_energy(seeded) < _scf_energy(plain) - 1e-3


def test_guess_chk_other_method_same_answer(tmp_path):
    chk = str(tmp_path / "hf.chk")
    run("minpop_rohf.py", "-xyz", f"{XYZ}/ch2.xyz", "-mult", "3", "-basis", "6-31G",
        "-chk", chk)
    base = ("-xyz", f"{XYZ}/ch2.xyz", "-mult", "3", "-basis", "6-31G*", "-no-chk")
    a = run("minpop_uks.py", *base)
    b = run("minpop_uks.py", *base, "-guess-chk", chk)
    assert abs(_scf_energy(a) - _scf_energy(b)) < 1e-7
    assert np.allclose(charges(a), charges(b), atol=1e-5)
    err = run("minpop_uks.py", "-xyz", f"{XYZ}/ch2.xyz", "-mult", "1", "-guess-chk", chk,
              ok=False)
    assert "multiplicity" in err


def test_guess_chk_rotated_frame():
    """Stored orbitals rotated into another frame reproduce the energy,
    spherical and Cartesian."""
    sys.path.insert(0, PKG)
    import minpop_chk
    from pyscf import gto, scf
    th = 0.6
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, np.cos(1.1), -np.sin(1.1)], [0, np.sin(1.1), np.cos(1.1)]])
    xyz = np.array([[0, 0, 0.1], [0, 0.8, -0.5], [0.3, -0.7, -0.4]])
    for cart in (False, True):
        mk = lambda x: gto.M(atom=[["O", x[0]], ["H", x[1]], ["H", x[2]]], basis="6-31g*",
                             charge=1, spin=1, cart=cart, verbose=0)
        chk = os.path.join(WORK, f"rot{int(cart)}.chk")
        mf = scf.UHF(mk(xyz))
        mf.chkfile = chk
        mf.conv_tol = 1e-10
        mf.kernel()
        mol2 = mk(xyz @ R.T + 0.4)
        dm = minpop_chk.guess_density(chk, mol2, "UHF", verbose=False)
        assert abs(scf.UHF(mol2).energy_tot(dm=dm) - mf.e_tot) < 1e-8
