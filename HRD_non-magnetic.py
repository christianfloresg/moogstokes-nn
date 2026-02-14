import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

# Folder containing the .iso files
folder = "data/isochrones"

# Pre-defined ages (Myr)
model_ages = [1, 2, 4, 8, 10, 15]

iso_data = {}  # {age: {"mass":[], "logTeff":[], "logg":[], "mconv_env":[]}}

# Read all matching files
for num in model_ages:
    fname_pattern = os.path.join(folder, f"dmestar_{float(num):07.1f}myr_z+0.00_a+0.00_phx.iso")
    files = glob.glob(fname_pattern)

    for fname in files:
        with open(fname, "r") as f:
            lines = f.readlines()
            # Extract age from the third commented line if needed
            header_line = lines[2]
            m = re.search(r"Age\s*=\s*([\d\.]+)\s*Myr", header_line)
            age = float(m.group(1)) if m else float(num)

        # Read numerical data (skip first 5 lines)
        arr = np.loadtxt(fname, skiprows=5)
        mass = arr[:, 0]
        logTeff = arr[:, 1]
        logg = arr[:, 2]
        mconv_env = arr[:, 6]

        iso_data.setdefault(age, {"mass": [], "logTeff": [], "logg": [], "mconv_env": []})
        iso_data[age]["mass"] = mass
        iso_data[age]["logTeff"] = 10**logTeff
        iso_data[age]["logg"] = logg
        iso_data[age]["mconv_env"] = mconv_env

# --- PLOT 1: Isochrones ---
plt.figure(figsize=(10, 8))
for age in sorted(iso_data.keys()):
    plt.plot(iso_data[age]["logTeff"], iso_data[age]["logg"], label=f"{age:.1f} Myr",color='k',linestyle='--')
plt.gca().invert_xaxis()  # Teff decreases to the right in HR diagrams
# plt.xlabel("log(Teff) [K]")
# plt.ylabel("log(g) [cm/s²]")
# plt.title("Isochrones in Teff–log(g) Plane")
# plt.legend()
# plt.grid(True)

# --- PLOT 2: Mass tracks ---
# plt.figure(figsize=(10, 8))
# Get unique masses from first age set
ref_age = sorted(iso_data.keys())[0]
unique_masses = [0.1,0.5,0.8,1.5,2.0,2.5] #iso_data[ref_age]["mass"]
#ratios_to_plot = [0.99, 0.60, 0.20]

for m in unique_masses:
    teffs, loggs = [], []
    for age in sorted(iso_data.keys()):
        idx = np.where(np.isclose(iso_data[age]["mass"], m, atol=0.01))[0]
        if len(idx) > 0:
            teffs.append(iso_data[age]["logTeff"][idx[0]])
            loggs.append(iso_data[age]["logg"][idx[0]])
    if teffs and loggs:
        plt.plot(teffs, loggs, marker=None, label=f"{m:.2f} Msun" if m in unique_masses else "",color='red')
plt.gca().invert_xaxis()
# plt.xlabel("log(Teff) [K]")
# plt.ylabel("log(g) [cm/s²]")
# plt.title("Mass Tracks in Teff–log(g) Plane")
# plt.legend()
# plt.grid(True)

# --- PLOT 3: Convective envelope ratio points ---
ratios_to_plot = [0.95, 0.66, 0.33, 0.10]
colors = ['r', 'g', 'b' , 'orange']

# plt.figure(figsize=(10, 8))
for age in sorted(iso_data.keys()):
    logTeff = iso_data[age]["logTeff"]
    logg = iso_data[age]["logg"]
    mass = iso_data[age]["mass"]
    mconv = iso_data[age]["mconv_env"]

    for ratio, c in zip(ratios_to_plot, colors):
        mask = np.isclose(mconv / mass, ratio, atol=0.015)
        plt.scatter(logTeff[mask], logg[mask], color=c,
                    label=f"{ratio:.2f}" if age == sorted(iso_data.keys())[0] else "")

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel("Teff [K]")
plt.ylabel("log(g) [cm/s²]")
plt.title("Convective Envelope Ratios in Teff–log(g) Plane")
plt.legend(title="Mconv/Mass")
plt.grid(True)
plt.xlim(8000,2000)

plt.show()
