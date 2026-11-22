#!/bin/bash
# CrH, CrO, Cr2 at experimental bond lengths: GVB-PP (keithgroup/gvbpp) followed
# by MinPop. ANO-RCC-VDZP, X2C, ANO-RCC-MB projection.
# Pair counts: CrH 1 (4s-1s sigma, five 3d open), CrO 4 (sigma, pi, two O lone
# pairs, four open), Cr2 6 (the sextuple bond). The pair search is restricted to
# the valence orbitals automatically; semicore 3s/3p never enter it.
# Cr2 has several perfect-pairing minima: run two seeds and keep the lowest.
set -u
export PYTHONIOENCODING=utf-8
OPTS="-basis ano-rcc-vdzp -minimal-basis ano-rcc-mb -x2c -symmetry-shells -soscf"
minpop-gvbpp -xyz xyz/crh.xyz -mult 6 -npairs 1 $OPTS                         > crh_gvb.out 2>&1
minpop-gvbpp -xyz xyz/cro.xyz -mult 5 -npairs 4 $OPTS                         > cro_gvb.out 2>&1
minpop-gvbpp -xyz xyz/cr2.xyz -mult 1 -npairs 6 $OPTS -gvb-trials 8 -gvb-seed 0 > cr2_gvb_s0.out 2>&1
minpop-gvbpp -xyz xyz/cr2.xyz -mult 1 -npairs 6 $OPTS -gvb-trials 8 -gvb-seed 1 > cr2_gvb_s1.out 2>&1
grep -H "converged GVB-PP energy\|WARNING" crh_gvb.out cro_gvb.out cr2_gvb_s?.out

# Checkpoints (README section 11). Each run above wrote <name>_gvbpp.chk; the two
# Cr2 seeds share cr2_gvbpp.chk, so name the checkpoint when you keep both:
#   ... -gvb-seed 0 -chk cr2_s0.chk      ... -gvb-seed 1 -chk cr2_s1.chk
# Reproject a stored GVB-PP wavefunction, no SCF and no GVB-PP optimization:
minpop-gvbpp -read-chk crh_gvbpp.chk -minimal-basis ano-rcc-mb-occ -symmetry-shells > crh_gvb_occ.out 2>&1
minpop-chk   cro_gvbpp.chk -minimal-basis sto-3g-ano -symmetry-shells              > cro_gvb_sto3g_ano.out 2>&1
minpop-chk   cro_gvbpp.chk -info
# Start a larger-basis GVB-PP from the stored natural orbitals (one optimization
# instead of the multi-start):
minpop-gvbpp -xyz xyz/crh.xyz -mult 6 -npairs 1 -basis ano-rcc-vtzp -minimal-basis ano-rcc-mb \
             -x2c -symmetry-shells -soscf -guess-chk crh_gvbpp.chk -chk crh_vtzp.chk > crh_gvb_vtzp.out 2>&1
grep -H "GVB-PP start\|converged GVB-PP energy" crh_gvb_vtzp.out
# GVB-PP on stored SCF orbitals: no SCF, the checkpoint's orbitals are the
# reference (ROHF here; RKS, UHF and UKS checkpoints work the same way):
minpop-rohf  -xyz xyz/crh.xyz -mult 6 -basis ano-rcc-vdzp -x2c -soscf -chk crh_rohf.chk > crh_rohf.out 2>&1
minpop-gvbpp -read-chk crh_rohf.chk -npairs 1 -minimal-basis ano-rcc-mb -symmetry-shells \
             -chk crh_from_rohf.chk > crh_gvb_from_rohf.out 2>&1
grep -H "GVB-PP reference\|converged GVB-PP energy" crh_gvb_from_rohf.out
