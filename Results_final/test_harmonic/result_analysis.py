#%%
import os 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

orb_Es_harmonic_oscillator = np.loadtxt(f"{SCRIPT_DIR}/orb_Es_harmonic_oscillator.csv", delimiter=",")
orb_Es_ref = np.loadtxt(f"{SCRIPT_DIR}/orb_Es_ref.csv", delimiter=",")

mps_es = pd.read_csv(f"{SCRIPT_DIR}/mps/max_bond_dim_5/lobpcg_energies.csv", delimiter=",",header=None)
threetree_es = pd.read_csv(f"{SCRIPT_DIR}/threetree/max_bond_dim_5/lobpcg_energies.csv", delimiter=",",header=None)
color_mps = plt.cm.Blues(np.linspace(0.3, 1.0, 30))
color_threetree = plt.cm.Oranges(np.linspace(0.3, 1.0, 30))
for i in range(30):
    plt.plot(range(1,7), abs(mps_es.iloc[i]-orb_Es_ref[i]), color=color_mps[i])
    plt.plot(range(1,7), abs(threetree_es.iloc[i]-orb_Es_ref[i]), color=color_threetree[i])

plt.plot(range(1,7), abs(mps_es.iloc[14]-orb_Es_ref[14]), label="MPS", color=color_mps[14], linewidth=2)
plt.plot(range(1,7), abs(threetree_es.iloc[14]-orb_Es_ref[14]), label="T3NS", color=color_threetree[14], linewidth=2)
plt.yscale("log")
plt.xlabel("Iteration", fontsize=18)
plt.ylabel("Absolute Error (cm-1)", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f"{SCRIPT_DIR}/plots/convergence_all.pdf")
plt.close()
# %%
mps_error = mps_es[5] - orb_Es_ref
threetree_error = threetree_es[5] - orb_Es_ref
plt.scatter(range(len(mps_error)), abs(mps_error), label="MPS", color=color_mps[15], marker="o")
plt.scatter(range(len(threetree_error)), abs(threetree_error), label="T3NS", color=color_threetree[15], marker="s")
plt.yscale("log")
plt.xlabel("State Index", fontsize=18)
plt.ylabel("Final Absolute Error (cm-1)", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f"{SCRIPT_DIR}/plots/final_error_comparison_harmonic.pdf")
plt.close()
# %%