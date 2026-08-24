#!/bin/bash
# Submit all 10 day/night ablation experiments (Large (192x192, 64ch))
set -e

bsub < lsf/昼夜消融/训练任务/abl_large_day_all.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_night_all.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_day_no_terrain.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_night_no_terrain.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_day_no_thermal.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_night_no_thermal.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_day_no_pblh.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_night_no_pblh.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_day_no_pressure.lsf
bsub < lsf/昼夜消融/训练任务/abl_large_night_no_pressure.lsf
