"""Generate individual LSF scripts for day/night ablation experiments."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LSF_DIR = os.path.join(BASE_DIR, "lsf", "ablation_day_night")

ABLATIONS = [
    ("all", "config_wind_3d_ablation_all"),
    ("no_terrain", "config_wind_3d_ablation_no_terrain"),
    ("no_thermal", "config_wind_3d_ablation_no_thermal"),
    ("no_pblh", "config_wind_3d_ablation_no_pblh"),
    ("no_pressure", "config_wind_3d_ablation_no_pressure"),
]

FILTERS = ["day", "night"]

LSF_TEMPLATE = """#!/bin/bash
# ============================================
# LSF — Day/Night Ablation: {abl_desc} ({filter_name})
# 128x128 + inner_ch=32 + pure L2 + GPU
# ============================================
#BSUB -q 723090ib
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -J {job_name}
#BSUB -o logs/{job_name}_%J.out
#BSUB -e logs/{job_name}_%J.err

module load anaconda/3
module load cuda/11.8.0
source activate wind3d

cd $LS_SUBCWD
mkdir -p logs

echo "=== Ablation: {abl_desc} | Filter: {filter_name} ==="
echo "Config: {config_path}"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

python scripts/train_schrodinger_bridge_model.py \\
  --config_path {config_path} \\
  --experiment_name ExperimentSchrodingerBridge3dWind \\
  --device cuda:0

echo ""
echo "End: $(date)"
"""

ABL_DESC = {
    "all": "All10 (baseline)",
    "no_terrain": "NoTerrain",
    "no_thermal": "NoThermal",
    "no_pblh": "NoPBLH",
    "no_pressure": "NoPSFC",
}


def main():
    os.makedirs(LSF_DIR, exist_ok=True)

    for abl_key, config_base in ABLATIONS:
        for filter_name in FILTERS:
            job_name = f"abl_{filter_name}_{abl_key}"
            config_path = f"configs/{config_base}_{filter_name}.yml"
            content = LSF_TEMPLATE.format(
                abl_desc=ABL_DESC[abl_key],
                filter_name=filter_name,
                job_name=job_name,
                config_path=config_path,
            )
            out_path = os.path.join(LSF_DIR, f"{job_name}.lsf")
            with open(out_path, "w") as f:
                f.write(content)
            print(f"Created: lsf/ablation_day_night/{job_name}.lsf")

    # Also create a submit-all shell script
    submit_all = os.path.join(LSF_DIR, "submit_all.sh")
    lines = ["#!/bin/bash", "# Submit all 10 day/night ablation experiments", "set -e", ""]
    for abl_key, _ in ABLATIONS:
        for filter_name in FILTERS:
            job_name = f"abl_{filter_name}_{abl_key}"
            lines.append(f"bsub < lsf/ablation_day_night/{job_name}.lsf")
    with open(submit_all, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(submit_all, 0o755)
    print(f"\nCreated: lsf/ablation_day_night/submit_all.sh")


if __name__ == "__main__":
    main()
