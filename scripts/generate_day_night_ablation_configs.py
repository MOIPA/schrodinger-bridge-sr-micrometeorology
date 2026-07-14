"""Generate day/night ablation configs from base ablation configs."""
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")

ABLATION_BASES = {
    "all": "config_wind_3d_ablation_all.yml",
    "no_terrain": "config_wind_3d_ablation_no_terrain.yml",
    "no_thermal": "config_wind_3d_ablation_no_thermal.yml",
    "no_pblh": "config_wind_3d_ablation_no_pblh.yml",
    "no_pressure": "config_wind_3d_ablation_no_pressure.yml",
}

FILTERS = ["day", "night"]


def main():
    for abl_name, base_file in ABLATION_BASES.items():
        base_path = os.path.join(CONFIG_DIR, base_file)
        with open(base_path) as f:
            config = yaml.safe_load(f)

        for filter_name in FILTERS:
            config["data"]["day_night_filter"] = filter_name
            out_name = f"config_wind_3d_ablation_{abl_name}_{filter_name}.yml"
            out_path = os.path.join(CONFIG_DIR, out_name)
            with open(out_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Created: {out_name}")


if __name__ == "__main__":
    main()
