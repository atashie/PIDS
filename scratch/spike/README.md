# DOLFINx install spike

Validates the Option-B FEM stack **on this machine** before we commit the forward-model architecture (per `DECISION-model-selection.md`, 2026-06-04, and the cautious-building guardrail). Throwaway — lives in `scratch/`, not the formal model code.

**What it proves:**
- **A.** DOLFINx + PETSc linear solve works (Poisson, manufactured solution → ~machine-zero error).
- **B.** PETSc variational-inequality solver (`vinewtonrsls`) works — the **exact** path the reverse PIDS catch-valve needs.

A `PASS` on both means the architecture is feasible here and we proceed to the design + module 1.

---

## Step 1 — install WSL2 (you, once; needs admin + a reboot)
WSL is **not** installed on this machine. In an **elevated** PowerShell (Run as administrator):

```powershell
wsl --install -d Ubuntu
```

Reboot when prompted, then launch **Ubuntu** from the Start menu and set a UNIX username/password. (You can run the command for me from this session with `! wsl --install -d Ubuntu`, but the elevation + reboot must happen on your side.)

## Step 2 — install conda + the environment (I can drive this once WSL is up)
Inside Ubuntu (WSL):

```bash
# Miniforge (conda-forge by default)
wget -qO- https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
bash /tmp/mf.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# the repo is visible from WSL under /mnt/c
cd /mnt/c/Users/arikt/Documents/GitHub/PIDS/scratch/spike
conda env create -f environment.yml      # (or: mamba env create -f environment.yml)
```

## Step 3 — run the spike
```bash
conda activate pids-fem
python smoke_test.py        # exit 0 + "SPIKE PASS" = good to go
```

---

**Tell me when WSL is installed** (after Step 1) and I'll drive Steps 2–3, read the output, and — on PASS — write the forward-model architecture design and start module 1 (the subsurface Richards solver), each gated by `governance/claude-sanity-check-routine.md`.
