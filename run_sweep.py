import numpy as np
import subprocess
import os
from tqdm import tqdm

def run_sweep():
    k_values = np.linspace(-5, 5, 20)
    beta_values = np.linspace(0, 1, 20)

    model_type = "mc"
    model_path = "model_weights/baseline_bftq_model.pt"
    num_envs = 8 
    n_episodes = 16
    env_id = "two-way-v0"

    print(f"Starting parameter sweep for {len(k_values)} k values and {len(beta_values)} beta values.")

    for k in tqdm(k_values, desc="k"):
        for beta in tqdm(beta_values, desc="beta", leave=True):
            command = [
                "python3",
                "inference.py",
                "--model-type", model_type,
                "--model-path", model_path,
                "--num-envs", str(num_envs),
                "--n-episodes", str(n_episodes),
                "--env", env_id,
                "--k", str(k),
                "--fixed-beta", str(beta)
            ]
            # print(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL)
            print(result.stdout)

    print("Parameter sweep completed.")

if __name__ == "__main__":
    run_sweep()
