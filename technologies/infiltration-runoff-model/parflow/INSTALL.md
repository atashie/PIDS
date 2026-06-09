# ParFlow — Install & Smoke Test (Docker-first, WSL2)

Reproducible recipe, **verified 2026-06-08** on WSL2 Ubuntu 26.04 LTS (codename
`resolute`), x86_64, with systemd as PID 1. All commands run *inside* WSL Ubuntu.

## 0. Prerequisites (confirmed on this machine)
- WSL2 + Ubuntu with **systemd enabled** (`/etc/wsl.conf` contains `[boot]` → `systemd=true`).
- Passwordless sudo (otherwise run the `sudo` steps yourself).
- Internet access (image pull ≈ 0.9 GB).
- Repo mounted at `/mnt/c/Users/arikt/Documents/GitHub/PIDS` (for reference; see the fs-speed note in §5).

## 1. Install Docker Engine (inside WSL Ubuntu)
We use Ubuntu's packaged engine (`docker.io`), **not** Docker's own apt repo: `docker.io`
always matches the installed Ubuntu release, whereas Docker's repo lagged on the brand-new
26.04 (`resolute`) codename. **Docker Desktop is not used** (lighter, no Desktop licensing,
native-Linux performance).

```bash
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
sudo systemctl enable --now docker            # start daemon via systemd
sudo usermod -aG docker $(whoami)             # optional: run docker without sudo
sudo docker run --rm hello-world              # verify the daemon
```
Installed here: **docker.io 29.1.3** (containerd 2.2.2, runc 1.4.0).

> The `docker` group only takes effect in a **fresh** WSL session: run `wsl --shutdown`
> from Windows, then reopen Ubuntu. Until then, prefix commands with `sudo docker`.

## 2. Pull the ParFlow image
```bash
sudo docker pull parflow/parflow:latest
```
- Official image (published by the ParFlow project), ≈ 0.9 GB.
- Digest pinned 2026-06-08: `sha256:ac194a197952c5801f5027d272679279e640bec7b20134fe14194247440b77b3`.
- Container wiring (from `docker inspect`): `ENTRYPOINT=[pfrun]`, `WORKDIR=/data`,
  `PARFLOW_DIR=/usr/opt/parflow`, MPI preset with `--allow-run-as-root --oversubscribe`.

## 3. The run idiom
```bash
sudo docker run --rm -v <host_dir>:/data parflow/parflow:latest <script>.{tcl|py} <Px> <Py> <Pz>
```
- `<host_dir>` is mounted to `/data` (the container workdir). **All input files must live in
  `<host_dir>`** (subdirectories are fine; symlinks do **not** work across the mount).
- `<Px> <Py> <Pz>` = MPI process topology along X/Y/Z. Use `1 1 1` for a serial run.
- File extension selects the engine: `.tcl` → `tclsh`, `.py` → pftools / Python.
- Bundled examples ship in the image at `/usr/opt/parflow/examples/`
  (`example_single.tcl`, `example_single.py`, `pftest.tcl`) with reference outputs in
  `/usr/opt/parflow/examples/correct_output/`. The pftools Python package source is at
  `/usr/opt/parflow/python/parflow`.

## 4. Smoke test (verified 2026-06-08)
```bash
mkdir -p $HOME/parflow-smoke
# copy a bundled example out of the image into the scratch dir:
sudo docker run --rm -v $HOME/parflow-smoke:/data --entrypoint cp \
  parflow/parflow:latest /usr/opt/parflow/examples/example_single.tcl /data/
# run it (serial):
sudo docker run --rm -v $HOME/parflow-smoke:/data \
  parflow/parflow:latest example_single.tcl 1 1 1
# verify clean completion:
grep -c "Problem solved" $HOME/parflow-smoke/example_single.out.txt   # -> 1
```
**Result:** `Problem solved`, no errors, Total Runtime 0.12 s. Output taxonomy produced:
- `*.out.press.*.pfb` — pressure-head field (primary state)
- `*.out.perm_{x,y,z}.pfb`, `*.out.porosity.pfb` — material fields
- `*.out.concen.*.pfsb` — transport concentration (this example exercises advection)
- `*.out.log`, `*.out.txt` — run logs (`Problem solved` marks success)
- `*.out.timing.csv` — per-timer performance breakdown (Solver, PCG, I/O, …)
- `*.pfidb`, `*.out.pftcl` — resolved input database

## 5. Gotchas / notes for later stages
- **Filesystem speed:** run cases on the **native WSL fs** (`$HOME/...`), *not* under
  `/mnt/c/...` (the repo). The Windows-mounted 9p filesystem is slow for ParFlow's many-file
  I/O. Keep curated *scripts* in this repo folder (`cases/`); run + write outputs under
  `$HOME`, then copy results back for archiving. (This matters for the performance comparison.)
- **File ownership:** the container runs as root, so host outputs are root-owned
  (world-readable). Use `sudo chown -R $(whoami) <dir>` if you need to edit/delete them.
- **Performance fairness (stage D):** the Dockerized engine carries a small overhead vs. a
  native build. Pin thread counts identically across both models, and re-confirm any headline
  performance numbers with a native ParFlow build before relying on them.
- **Reading outputs:** `.pfb`/`.pfsb` are ParFlow binary; read them into numpy with pftools
  (`from parflow import read_pfb`) for comparison against the in-house model's NetCDF.

## 6. Uninstall (if ever needed)
```bash
sudo docker image rm parflow/parflow:latest
sudo apt-get remove -y docker.io        # removes the engine; add --purge to drop config
```
