#!/bin/bash
# Submit all 10 day/night ablation experiments
set -e

bsub < lsf/昼夜消融/训练任务/abl_day_all.lsf
bsub < lsf/昼夜消融/训练任务/abl_night_all.lsf
bsub < lsf/昼夜消融/训练任务/abl_day_no_terrain.lsf
bsub < lsf/昼夜消融/训练任务/abl_night_no_terrain.lsf
bsub < lsf/昼夜消融/训练任务/abl_day_no_thermal.lsf
bsub < lsf/昼夜消融/训练任务/abl_night_no_thermal.lsf
bsub < lsf/昼夜消融/训练任务/abl_day_no_pblh.lsf
bsub < lsf/昼夜消融/训练任务/abl_night_no_pblh.lsf
bsub < lsf/昼夜消融/训练任务/abl_day_no_pressure.lsf
bsub < lsf/昼夜消融/训练任务/abl_night_no_pressure.lsf
