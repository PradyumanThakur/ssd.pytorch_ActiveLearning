"""
Experiment Manager for Active Learning

Responsible for
----------------
1. Creating experiment directory structure
2. Managing AL rounds
3. Managing paths
4. Saving configs and history
"""

from pathlib import Path
import os
import json
import yaml
import pandas as pd


class ExperimentManager:
    def __init__(self, experiment_name: str, root_dir: str = "experiments"):
        self.root = Path(root_dir)
        self.exp_dir = self.root / experiment_name
    
    def create_experiment(self):
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        for folder in ["initial", "random", "cdal"]:
            (self.exp_dir / folder).mkdir(exist_ok=True)

    def create_initial_round(self):
        self._create_round(self.exp_dir / "initial" / "round_00")

    def create_method_round(self, method, round_id):
        round_dir = (self.exp_dir / method / f"round_{round_id:02d}")
        self._create_round(round_dir)

    def _create_round(self, round_dir):
        folders = ["splits", "checkpoints", "inference", "selected", "evaluation"]
        round_dir.mkdir(parents=True, exist_ok=True)

        for folder in folders:
            (round_dir / folder).mkdir(exist_ok=True)
        
    def save_config(self, config, path):
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
    def update_history(self, csv_path, row):
        df = pd.DataFrame([row])
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
            return

        history = pd.read_csv(csv_path)

        mask = (
            (history["method"] == row["method"]) &
            (history["round"] == row["round"])
        )

        if mask.any():
            history.loc[mask, :] = row
        else:
            history = pd.concat([history, df], ignore_index=True)

        history.to_csv(csv_path, index=False)
    
    def get_round_dir(self, method, round_id):
        if method == "initial":
            return (self.exp_dir / "initial" / "round_00")
        
        return (self.exp_dir / method / f"round_{round_id:02d}")
    
    def get_split_dir(self, method, round_id):
        return (self.get_round_dir(method, round_id) / "splits")

    def get_weight_dir(self, method, round_id):
        return (self.get_round_dir(method, round_id) / "checkpoints")

    def get_inference_dir(self, method, round_id):
        return (self.get_round_dir(method, round_id) / "inference")

    def get_selected_dir(self, method, round_id):
        return (self.get_round_dir(method, round_id) / "selected")

    def get_evaluation_dir(self, method, round_id):
        return (self.get_round_dir(method, round_id) / "evaluation")
    
    @property
    def experiment_config(self):
        return self.exp_dir / "experiment_config.yaml"

    @property
    def al_history(self):
        return self.exp_dir / "al_history.csv"

    @property
    def per_class_ap(self):
        return self.exp_dir / "per_class_ap.csv"