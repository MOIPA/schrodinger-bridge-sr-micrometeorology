"""Generate day/night ablation configs (small or large model)."""
import argparse
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
LSF_DIR = os.path.join(BASE_DIR, "lsf", "ablation_day_night")

ABLATION_BASES = {
    "all": "config_wind_3d_ablation_all.yml",
    "no_terrain": "config_wind_3d_ablation_no_terrain.yml",
    "no_thermal": "config_wind_3d_ablation_no_thermal.yml",
    "no_pblh": "config_wind_3d_ablation_no_pblh.yml",
    "no_pressure": "config_wind_3d_ablation_no_pressure.yml",
}

ABL_DESC = {
    "all": "All10",
    "no_terrain": "NoTerrain",
    "no_thermal": "NoThermal",
    "no_pblh": "NoPBLH",
    "no_pressure": "NoPSFC",
}

FILTERS = ["day", "night"]

LSF_TEMPLATE = """#!/bin/bash
# ============================================
# LSF — Day/Night Ablation {size_label}: {abl_desc} ({filter_name})
# {resolution} + inner_ch={inner_ch} + pure L2 + GPU
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

echo "=== Ablation: {abl_desc} | Filter: {filter_name} | {size_label} ==="
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true",
                        help="Generate large model configs (192x192, inner_ch=64)")
    args = parser.parse_args()

    if args.large:
        resolution = 192
        inner_ch = 64
        epochs = 600
        suffix = "_large"
        size_label = "Large (192x192, 64ch)"
        job_prefix = "abl_large_"
    else:
        resolution = 128
        inner_ch = 32
        epochs = 400
        suffix = ""
        size_label = "Small (128x128, 32ch)"
        job_prefix = "abl_"

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LSF_DIR, exist_ok=True)

    for abl_name, base_file in ABLATION_BASES.items():
        base_path = os.path.join(CONFIG_DIR, base_file)
        with open(base_path) as f:
            config = yaml.safe_load(f)

        for filter_name in FILTERS:
            # Update model params
            config["data"]["hr_cropped_shape"] = [resolution, resolution]
            config["data"]["hr_data_shape"] = [resolution, resolution]
            config["data"]["day_night_filter"] = filter_name
            config["model"]["inner_channel"] = inner_ch
            config["train"]["epochs"] = epochs

            out_name = "config_wind_3d_ablation_{}_{}{}.yml".format(
                abl_name, filter_name, suffix)
            out_path = os.path.join(CONFIG_DIR, out_name)
            with open(out_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
            print("Config: {}".format(out_name))

            # Generate LSF script
            job_name = "{}{}_{}".format(job_prefix, filter_name, abl_name)
            config_path = "configs/" + out_name
            lsf_content = LSF_TEMPLATE.format(
                size_label=size_label,
                abl_desc=ABL_DESC[abl_name],
                filter_name=filter_name,
                resolution="{}x{}".format(resolution, resolution),
                inner_ch=inner_ch,
                job_name=job_name,
                config_path=config_path,
            )
            lsf_path = os.path.join(LSF_DIR, "{}.lsf".format(job_name))
            with open(lsf_path, "w") as f:
                f.write(lsf_content)
            print("  LSF:  lsf/ablation_day_night/{}.lsf".format(job_name))

    # Submit-all script
    submit_all = os.path.join(LSF_DIR, "submit_all{}.sh".format(suffix))
    lines = ["#!/bin/bash",
             "# Submit all 10 day/night ablation experiments ({})".format(size_label),
             "set -e", ""]
    for abl_name in ABLATION_BASES:
        for filter_name in FILTERS:
            job_name = "{}{}_{}".format(job_prefix, filter_name, abl_name)
            lines.append("bsub < lsf/ablation_day_night/{}.lsf".format(job_name))
    with open(submit_all, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(submit_all, 0o755)
    print("\nSubmit-all: lsf/ablation_day_night/submit_all{}.sh".format(suffix))


if __name__ == "__main__":
    main()
