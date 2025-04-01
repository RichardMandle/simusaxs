# simuSAXS

_A python tool for simulating and small angle X-ray scattering patterns based on MD trajectories_

---

### How?

This code takes a molecular dynamics trajectory (e.g., from GROMACS) and does a few things:

- Loads and aligns the system along its principal axis of orientation (e.g. the director, for a liquid crystal)
- Builds 3D electron density maps from atomic coordinates using radii from Bondii
- Computes the 3D structure factor via FFT
- Generates nice 2D SAXS images; these are projections (i.e. summations) of the 3DSF along an axis of some sort (with optional radial integration)
- Supports optional band-pass filtering, resolution tuning, interpolation, and more

The code is designed to be fast (Numba + FFTW) and flexible - there are lots of ways to control the output.

Depending on your simulation and the settings you use, the ammount of RAM required can be _very large_. I'll look at fixing this in future.

Want to see it in use? https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202416545 (arxiv: https://arxiv.org/abs/2408.15859)

---

### Quick Start

```bash
python simusaxs.py -top system.gro -traj traj.trr -o output_name
```

This will generate the 3D structure factor data and PNG plots of the XY/XZ/YZ projections.

You can also reload saved .npz files to skip the heavy lifting:

```
python simusaxs.py -rld output_name
```

### Useful Options
Option	What it does
-qmin / -qmax	Set Q-space range (nm⁻¹)
-r	Resolution in nm (default = 0.05)
-int	Turn on interpolation of SF
-plt	Generate radial profiles from SAXS images
-lines	Overlay Q-circles and labels on plots
-save	Save edensity and SF to .npz for later
-quietly	Suppress verbose output
-pad	Padding size for FFT (default = 20)
-cmap	Change plot colormap (default = plasma)
Check out python simusaxs.py -h for a full list.

### Trajectory Requirements
Trajectories should be readable via MDTraj. The code will work with any mdtraj object, but it might need adjusting if you plan to give it anything other than .trr (I haven't really checked or looked).

You must provide both -traj and -top unless reloading

### Output
For each direction (xy, yz, xz), you get:

outputname_xy.png — linear scale image
outputname_xy_log.png — log scale image
outputname_xy.csv — raw data

Optional: outputname_xy_integrated.png and _log.png (radial integration)

Also saves:

outputname_edensity.npz
outputname_sf.npz

### Dependencies
numpy, scipy, matplotlib
pyfftw (for fast 3D FFTs)
mdtraj (for reading MD files)
numba (for JIT speedups)
tqdm, psutil, etc.


### Tips
If you're running on HPC: the script switches matplotlib to AGG to avoid display errors.
You can adjust Q-step size manually with -stp, or let it auto-pick based on Q-range.
If RAM is tight, consider reducing resolution or using fewer frames.
Typically using the normal resolution and a larger value of padding gives better results than setting arbitrarily large resolutions.
Setting resolution via -res controls the resoltion of the _real_ space electron grid; if you want to control the resolution of the _reciprocal_ space SAXS projection, you should use -pad 

### Real Workflow
First, make molecules whole across the PBC using gromacs:
```
gmx trjconv -f traj.trr -s testbox.tpr -o simusaxs_traj.trr -pbc whole
```
Next, pass this to simusaxs. We'll use a ```-pad``` of 100 to improve the smoothness of the resulting SAXS pattern
```
python $HOME/py_files/simusaxs/simusaxs.py -top confout.gro -traj simusaxs_traj.trr -pad 100
```

### Future Plans
Reduce memory use by chunking the trajectory
Calculate the SF per frame, but only store its cumulative total

### Credit
Written by R.J.M @ University of Leeds, 2023/2024
This code is a work in progress
