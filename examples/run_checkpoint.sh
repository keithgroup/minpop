#!/bin/bash
# One SCF, several minimal-basis projections (checkpoint files, README section 11).
# PtH doublet, UKS/PBE, ANO-RCC-VDZP on Pt and 6-31G* on H, atomic X2C.
set -u
export PYTHONIOENCODING=utf-8
XYZ=xyz/pth.xyz

# 1. SCF once. The converged wavefunction goes to pth.chk.
minpop-uks -xyz $XYZ -mult 2 -xc pbe -basis ano-rcc-vdzp -basis-light "6-31G*" \
           -x2c-atomic -minimal-basis sto-3g-ano -symmetry -chk pth.chk \
           -json pth_sto3g_ano.json > pth_sto3g_ano.out 2>&1

# 2. What the checkpoint holds.
minpop-chk pth.chk -info

# 3. Reproject: no SCF, seconds each. Either command form works.
minpop-uks -read-chk pth.chk -minimal-basis sto-3g-ano-occ -symmetry \
           -json pth_sto3g_ano_occ.json > pth_sto3g_ano_occ.out 2>&1
minpop-chk pth.chk -minimal-basis ano-rcc-mb -symmetry \
           -json pth_anorccmb.json > pth_anorccmb.out 2>&1
minpop-chk pth.chk -minimal-basis ano-rcc-mb-occ -symmetry-shells \
           -json pth_anorccmb_occ.json > pth_anorccmb_occ.out 2>&1

# 4. A new SCF started from the stored one: larger Pt basis, same state.
minpop-uks -xyz $XYZ -mult 2 -xc pbe -basis ano-rcc-vtzp -basis-light "6-31G*" \
           -x2c-atomic -minimal-basis sto-3g-ano -guess-chk pth.chk -chk pth_vtzp.chk \
           > pth_vtzp.out 2>&1
grep "Initial guess" pth_vtzp.out

for f in pth_sto3g_ano pth_sto3g_ano_occ pth_anorccmb pth_anorccmb_occ; do
  echo "== $f"
  grep -m1 "Checkpoint file:" $f.out
  grep -m1 "Minimal-basis projection" $f.out
  grep -A3 "MBS Mulliken charges and spin" $f.out | tail -2
done
