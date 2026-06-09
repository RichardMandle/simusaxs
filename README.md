# simuSAXS

_A python tool for simulating and small angle X-ray scattering patterns based on MD trajectories_

---

### How?

This code takes a molecular dynamics trajectory (e.g., from GROMACS) and does a few things:

- Loads and aligns the system along its principal axis of orientation (e.g. the director, for a liquid crystal)
- On a frame-by-frame basis, build a 3D map of electron density using Bondii atomic radii
- Computes the 3D structure factor via FFT and padding. These are summed, then windowed, then the accumulated total normalised. So its fairly RAM efficient.
- Generates nice 2D SAXS images; these are projections (i.e. summations) of the 3DSF along an axis of some sort (with optional radial integration)
- Supports optional zero-padding, band-pass filtering, resolution tuning, interpolation, and more

The code is designed to be fast (Numba + FFTW) and flexible - there are lots of ways to control the output, and produce publication-quality simulated SAXS data from MD trajectories.

You can speed up the analysis (although its already pretty fast) by using the *new* multithreaded mode (```-nt```) and specifying the number of CPU threads to use. A rough guide using timings for the structure factor calculation (the most intensive part):  

| ```-nt``` | walltime | speedup |
|-----------|----------|---------|
| 1 | 95.0 s | x1.0 |
| 2 | 53.0 s | x1.8 | 
| 3 | 41.0 s | x2.3 |
| 4 | 33.0 s | x2.9 |
| 8 | 28.0 s | x3.4 |

These done with a M5 processor so the drop-off in speedup between 4 threads and 8 threads probaby reflects the archetecture of the CPU somewhat. 130 frame trajectory with a 54k atom topology, and ```-pad``` set to 50. The plots look pretty nice, for example:

<img width="678" height="659" alt="image" src="https://github.com/user-attachments/assets/da319657-b57f-4950-a635-bfcbc0ad7078" />

The RAM usage of this newer version is not as bad (the above example had an estimated useage of ~1.5GB), but for large zero-paddings it can still easily dwarf what you have on hand, so its often best to resort to running on HPC. 

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
There are of course some other options you might need.

### Most Useful Options
|Option | What it does |
| ----- | ------------ |
| -pad  | Controls zero padding of the structure-factor; meaning you control resolution of the RECIPROCAL SPACE GRID. <br> Bigger value, smoother data (~ 100 is a good start). <br> Can use a lot of RAM. |
| -cmap | Change the colouring. Use a _perceptually uniform colourmap_ (plasma, magma, viridis, inferno etc) |
| -nt   | Control the number of CPU threads used for the FFT calculation (and only the FFT calculation). |
| -win  | Turn off FFT window filtering

### Less Useful Options
|Option	   |What it does|
| -------- | ---------- |
|-qmin / -qmax |	Set Q-space range (nm⁻¹) |
|-vox | Select size of voxels used for the electron density grid (in nm; default = 0.05) Note, for higher resolution, see ```-pad```  |
|-int |	Turn on interpolation of SF |
|-plt	| Generate radial profiles from SAXS images |
|-lines	| Overlay Q-circles and labels on plots |
|-save	| Save edensity and SF to .npz for later |
|-quietly	| Suppress verbose output |

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

For production work you can use larger values of ```-pad``` on your HPC, but the RAM requirements can become extreme.


### Credit
Written by R.J.M @ University of Leeds, 2023-2026
This code is a work in progress
