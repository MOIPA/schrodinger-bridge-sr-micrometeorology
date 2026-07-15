#!/bin/bash
# Submit all 10 day/night ablation experiments (Large (192x192, 64ch))
set -e

bsub < lsf/ablation_day_night/abl_large_day_all.lsf
bsub < lsf/ablation_day_night/abl_large_night_all.lsf
bsub < lsf/ablation_day_night/abl_large_day_no_terrain.lsf
bsub < lsf/ablation_day_night/abl_large_night_no_terrain.lsf
bsub < lsf/ablation_day_night/abl_large_day_no_thermal.lsf
bsub < lsf/ablation_day_night/abl_large_night_no_thermal.lsf
bsub < lsf/ablation_day_night/abl_large_day_no_pblh.lsf
bsub < lsf/ablation_day_night/abl_large_night_no_pblh.lsf
bsub < lsf/ablation_day_night/abl_large_day_no_pressure.lsf
bsub < lsf/ablation_day_night/abl_large_night_no_pressure.lsf
