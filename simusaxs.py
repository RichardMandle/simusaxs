#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pyfftw
import os
from numba import njit
import psutil 
import platform
import mdtraj as md
from string import digits
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import argparse

'''
part1 - elementary math stuff
'''

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
    
def rotation_matrix_from_vectors(vec1, vec2):
    '''
    Takes two input vectors (vec1, vec2) and returns the 
    rotation matrix required to transform vec1 into vec2. 
    
    Typical use would be that vec1 is the nematic director, and we
    reorient so its parallel to vec2.
    
    usage:
    otation_matrix_from_vectors(np.mean(director,axis=0),[1,0,0])
    '''
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix 

def get_radii(filepath='atomic_radii.txt'):
    if platform.system() == 'Linux':
        # Use $HOME environment variable to avoid hardcoding paths
        home_dir = os.path.expandvars('$HOME')
        filepath = os.path.join(home_dir, 'py_files', 'simusaxs', filepath)
    
    if platform.system() == 'Darwin':
        # Annoying its storred somewhere else on my laptop, so do this
        home_dir = os.path.expandvars('$HOME')
        filepath = os.path.join(home_dir, 'code', 'simusaxs', filepath)
    
    radii_dict = {}

    try:
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split('/')
                if len(parts) < 3:
                    raise ValueError(f"Invalid line format in {filepath}: {line}")

                atomic_number = int(parts[0].strip())
                atomic_symbol = parts[1].strip()
                try:
                    atomic_radius = float(parts[2].strip())  # Allow floating-point radii
                except ValueError:
                    raise ValueError(f"Invalid radius value in {filepath}: {parts[2]}")

                radii_dict[atomic_symbol] = {
                    'element': atomic_symbol,
                    'number': atomic_number,
                    'radius': atomic_radius
                }
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {filepath} does not exist.")
    except ValueError as ve:
        raise ValueError(f"Data error in {filepath}: {str(ve)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading {filepath}: {str(e)}")

    return radii_dict
    
'''
part 2 - OS type stuff
'''
def get_platform():
    '''
    Return information about the platform we are running on (incase of posix/NT differences)
    '''
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "separator": "\\" if platform.system() == "Windows" else "/"
    }
    if platform.system() == 'Linux':
        import matplotlib
        matplotlib.use('Agg') # assume headless HPC usage; use AGG for plotting.
    
    if args.quietly == False:
        print(f"OS:         {system_info['system']}")
        print(f"Release:    {system_info['release']}")
        print(f"Version:    {system_info['version']}")
    return system_info

    
'''
part 3 - MD trajectory stuff
'''

def traj_load(traj, top):
    '''
    Simply loads the trajectory and topology using mdtraj; seperate function for readability in code.
    '''
    return md.load(traj, top=top)

def get_director(traj, unit_vector=False):
    '''
    Get the nematic director in all frames and return it.
    '''
    mol_indices = [[n+x for x in range(int(traj.n_atoms/traj.n_residues))] for n in range(0, traj.n_atoms, int(traj.n_atoms/traj.n_residues))]
    director = np.mean(md.compute_directors(traj, mol_indices), 1)
    if unit_vector:
        director = np.array([unit_vector(d) for d in director])
    return director

import warnings

def compute_rescaling_factors(traj, warn_threshold=0.03):
    """
    compute rescaling factors based on the maximum dimension of the unit cell.

    NEW - throw an errorany axis needs >3% rescaling.
    """

    max_lengths = np.max(traj.unitcell_lengths, axis=0)
    min_lengths = np.min(traj.unitcell_lengths, axis=0)
    mean_lengths = np.mean(traj.unitcell_lengths, axis=0)

    rescaling_factors = max_lengths / traj.unitcell_lengths

    dimensions = ['X', 'Y', 'Z']

    if args.quietly == False:
        print('-' * 50)
        print('Unit-cell rescaling factors:')

        for i, dim in enumerate(dimensions):
            mean_rescale = np.mean(rescaling_factors[:, i])
            max_rescale = np.max(rescaling_factors[:, i])
            box_range = (max_lengths[i] - min_lengths[i]) / mean_lengths[i]

            print(
                f'{dim}: '
                f'mean rescale = {(mean_rescale - 1) * 100:6.2f} %, '
                f'max rescale = {(max_rescale - 1) * 100:6.2f} %, '
                f'box range = {box_range * 100:6.2f} %'
            )

            if (max_rescale - 1.0) > warn_threshold:
                warnings.warn(
                    f'Large {dim}-axis rescaling detected: '
                    f'max rescale = {(max_rescale - 1) * 100:.1f} %. '
                    f'This may indicate significant box drift, poor equilibration, '
                    f'or a trajectory segment spanning structural relaxation.'
                )
        print('\n')

    return rescaling_factors
    
def apply_rescaling(traj, rescaling_factors):
    '''
    Apply rescaling factors to the coordinates of a trajectory in-place.
    '''
    
    traj.xyz *= rescaling_factors[:, np.newaxis, :]
    traj.unitcell_lengths *= rescaling_factors[:, :]
    return traj

def rotate_trajectory(traj, rotation_matrix):
    '''
    Rotate the coordinates of a trajectory using a rotation matrix in-place.
    '''
    
    traj.xyz = np.einsum('ij,nkj->nki', rotation_matrix, traj.xyz)

def get_atom_information(traj):
    '''
    gets atom types from the trajectory
    
    which (will) contains atom names, the corresponding radius,
    the occupied volume, and the electron density within said volume
    
    '''
    atom_names = ([str(atom).split("-")[1] for atom in traj[0].topology.atoms])
    
    radii = get_radii() # load radii data
    
    atom_radii = []
    atom_volume = []
    atom_electrons = []
    
    for n in range(len(atom_names)):
        atom_names[n] = str(atom_names[n]).translate(str.maketrans('','',digits))
        atom_radii.append(radii[atom_names[n]]['radius']/1000)
        atom_volume.append((4/3) * atom_radii[n] ** 3)
        atom_electrons.append(radii[atom_names[n]]['number'])
    
    return(atom_names,atom_radii,atom_volume,atom_electrons)

def prepare_trajectory(traj,new_orientation=np.array([0,1,0])):
    '''
    Wrapper that prepares a trajectory for SAXS computation by performing multiple functions.
    '''
    
    if args.quietly == False:
        print('-'*50)
        print('preparing trajectory for SAXS computation')
    
    traj = apply_rescaling(traj, compute_rescaling_factors(traj))
    rotate_trajectory(traj, rotation_matrix_from_vectors(np.mean(get_director(traj),axis=0), new_orientation))
    traj.center_coordinates()
    
    return traj

def load_and_process_traj(traj, top):
    '''
    Wrapper function - loads a traj and gets it ready for analysis
    '''
    traj = traj_load(traj,top)
    traj = prepare_trajectory(traj)
    return(traj)

@njit
def update_electron_density(electron_density, center, gaussian_density, grid_size, grid_shape):
    ix0, ix1 = center[0] - grid_size // 2, center[0] + grid_size // 2
    iy0, iy1 = center[1] - grid_size // 2, center[1] + grid_size // 2
    iz0, iz1 = center[2] - grid_size // 2, center[2] + grid_size // 2
    
    i_modulo = np.arange(ix0, ix1) % grid_shape[0]
    j_modulo = np.arange(iy0, iy1) % grid_shape[1]
    k_modulo = np.arange(iz0, iz1) % grid_shape[2]
    
    flat_electron_density = electron_density.ravel()
    flat_gaussian_density = gaussian_density.ravel()
    grid_stride = grid_shape[1] * grid_shape[2]
    
    for i, i_wrap in enumerate(i_modulo):
        for j, j_wrap in enumerate(j_modulo):
            for k, k_wrap in enumerate(k_modulo):
                index_electron = i_wrap * grid_stride + j_wrap * grid_shape[2] + k_wrap
                index_gaussian = i * grid_size * grid_size + j * grid_size + k
                flat_electron_density[index_electron] += flat_gaussian_density[index_gaussian]

@njit
def create_indices(grid_size):
    indices = np.empty((3, grid_size, grid_size, grid_size), dtype=np.int32)
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                indices[0, i, j, k] = i
                indices[1, i, j, k] = j
                indices[2, i, j, k] = k
    return indices

@njit
def sum_squared_distances(grid_indices, grid_size):
    distances_sq = np.zeros_like(grid_indices[0])
    half_grid = grid_size // 2
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                distances_sq[i, j, k] = ((i - half_grid) ** 2 + (j - half_grid) ** 2 + (k - half_grid) ** 2)
    return distances_sq

@njit
def njit_minmax(x, fastmath=True):
    '''
    idea taken from:
    https://stackoverflow.com/questions/12200580/numpy-function-for-simultaneous-max-and-min
    
    gets the min/max value of a 3D array quicker than two seperate np.amin / np.amax calls.
    '''

    n_frames, n_atoms, n_coords = x.shape
    
    mini = np.copy(x[0, 0, :])
    maxi = np.copy(x[0, 0, :])
    
    for f in range(n_frames):
        for a in range(n_atoms):
            if f == 0 and a == 0:
                continue
            for c in range(n_coords):
                val = x[f, a, c]
                if val > maxi[c]:
                    maxi[c] = val
                elif val < mini[c]:
                    mini[c] = val
    return mini, maxi
    
@njit
def accumulate_power_spectrum(ft_flat, sf_flat):
    '''
    idea: accumulate fft power into flat array (sf_flat) without the full 3D array 
    when cmputing the structure factor

    ft_flat : flattened complex FFT output
    sf_flat : flattened float32 accumulator
    '''
    for i in range(ft_flat.size):
        re = ft_flat[i].real
        im = ft_flat[i].imag
        sf_flat[i] += re * re + im * im

def band_pass_filter_q(shape, delta_x, qmin=None, qmax=None):
    """
    new band pass filter intended to properly work in Q space

    shape   : tuple (Nx, Ny, Nz) of the padded grid
    delta_x : array-like of voxel sizes [dx, dy, dz] in nm
    qmin    : lower cutoff in nm^-1 (None = no low-Q cut)
    qmax    : upper cutoff in nm^-1 (None = no high-Q cut)
    """
    # 1D Q-axes from FFT: q = 2*pi * k, with k = fftfreq / d
    qx = np.fft.fftfreq(shape[0], d=delta_x[0]) * 2 * np.pi
    qy = np.fft.fftfreq(shape[1], d=delta_x[1]) * 2 * np.pi
    qz = np.fft.fftfreq(shape[2], d=delta_x[2]) * 2 * np.pi
    
    # Do it as float32 to save memory
    Qx, Qy, Qz = np.meshgrid(
        qx.astype(np.float32),
        qy.astype(np.float32),
        qz.astype(np.float32),
        indexing='ij'
    )

    qmag = np.sqrt(Qx*Qx + Qy*Qy + Qz*Qz, dtype=np.float32)

    del Qx, Qy, Qz

    mask = np.ones_like(qmag, dtype=bool)
    if qmin is not None:
        mask &= (qmag >= qmin)
    if qmax is not None:
        mask &= (qmag <= qmax)

    # as we will apply this to an fftshifted SF, we have to shift the mask too
    return np.fft.fftshift(mask)

def rough_ram_estimate_gb(traj, grid_shape, padded_shape, args, overestimate_factor = 1.2):
    '''
    New 09/06/2026
    new rough RAM estimator. Old one was just way off.
    
    deliberately overestimates a bit.
    '''

    N = np.prod(grid_shape, dtype=np.int64)
    Np = np.prod(padded_shape, dtype=np.int64)

    traj_gb = traj.xyz.nbytes / 1024**3 if hasattr(traj, "xyz") else 0.0

    grid_gb = (
        4 * N +    # electron_density float32
        4 * Np +   # padded density float32
        4 * Np +   # sf_accum float32
        8 * Np * 3 # complex FFT + copies/temporaries, rough complex64 estimate
    ) / 1024**3

    # band pass/filter/window overhead **rough** estimate
    overhead_gb = 0.0
    if args.fft_window:
        overhead_gb += (8 * N) / 1024**3

    if (args.qmin_filt is not None) or (args.qmax_filt is not None):
        overhead_gb += (32 * Np) / 1024**3

    total_gb = overestimate_factor * (traj_gb + grid_gb + overhead_gb)

    return total_gb
    
def make_gaussian_kernel(atom_radius, electrons, resolution, n_sigma):
    '''
    NEW 09/06/2026
    compute the atom electron density profiles seperately, out-of-the-loop
    to speed things up a bit (becasue we just end up calculating the same things
    over and over and over...)
    '''
    
    g_width = int(n_sigma * atom_radius / resolution)
    g_width = max(g_width, 2)

    half = g_width // 2
    x = np.arange(g_width, dtype=np.float32) - half
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = X*X + Y*Y + Z*Z

    sigma = atom_radius / resolution
    kernel = electrons / (sigma**3) * np.exp(-0.5 * r2 / sigma**2)

    return g_width, kernel.astype(np.float32)

def compute_3d_sf(traj):
    """
    Computes the 3D electron density and 3D structure factor from a traj object.

    Returns:
        electron_density : last-frame electron density (3D)
        sf_avg           : 3D structure factor averaged over frames (float32)
        q_axes           : tuple of 1D arrays (qx, qy, qz) in nm^-1 (fftshifted)
    """
    if args.n_sigma < 2:
        print('Adjusting Gaussian sigma value to 2 (you specified ' + str(args.n_sigma) + ', which would return an error)')
        args.n_sigma = 2

    _, atom_radii, _, atom_electrons = get_atom_information(traj)
    
    global_min, global_max = njit_minmax(traj.xyz)
    
    simulation_dimensions = global_max - global_min  # nm

    # real-space grid
    grid_shape = np.ceil(simulation_dimensions / args.voxel_size).astype(int)
    delta_x = simulation_dimensions / grid_shape  # actual voxel size in nm

    padded_shape = np.array(grid_shape) + 2 * args.padding
    padded_prod = np.prod(padded_shape, dtype=np.int64)

    available_gb = psutil.virtual_memory().available / 1024**3
    needed_gb = rough_ram_estimate_gb(traj, grid_shape, padded_shape, args)
    q_fund = 2.0 * np.pi / simulation_dimensions
    q_nyquist = np.pi / delta_x
    q_nyquist_limiting = np.min(q_nyquist)

    if not args.quietly:
        print('Requested vs Actual Simulation Dimensions')
        print('-'*50)
        print(f'Simulation Dimensions:  {simulation_dimensions} nm')
        print(f'Grid Shape:             {grid_shape}')
        print(f'Requested voxel size:   {args.voxel_size} nm')
        print(f'Actual voxel size:      {delta_x} nm')
        print(f"Q fundamental, 2π/Lbox: [{q_fund[0]:.4f}, {q_fund[1]:.4f}, {q_fund[2]:.4f}] nm^-1")
        print(f"Q Nyquist, π/dx:        [{q_nyquist[0]:.3f}, {q_nyquist[1]:.3f}, {q_nyquist[2]:.3f}] nm^-1")
        print(f"Limiting usable Q(max): {q_nyquist_limiting:.3f} nm^-1")
        print("\nRAM estimate")
        print("-" * 50)
        print(f"Estimated peak RAM needed :  {needed_gb:.1f} GB")
        print(f"RAM currently available   :  {available_gb:.1f} GB")

        if needed_gb > available_gb:
            print("WARNING: probably not enough RAM.")
        else:
            print(f"Plenty RAM there. Headroom:  {available_gb / needed_gb:.1f}x")

        print('-'*50)
        print('\n')
        
    if args.qmax is not None and args.qmax > q_nyquist_limiting:
        warnings.warn(
            f"Requested qmax={args.qmax:.3f} nm^-1 exceeds the limiting "
            f"Nyquist q={q_nyquist_limiting:.3f} nm^-1 from voxel size {delta_x}. "
            f"Use a smaller -qmax or a smaller -vox/--voxel_size."
        )
    
    electron_density = np.zeros(shape=grid_shape, dtype=np.float32)

    # acca for |F(q)|^2 on padded grid
    sf_accum = np.zeros(shape=padded_shape, dtype=np.float32)

    # build Q-axes in nm^-1, **fftshifted to match fftshifted SF**
    # NOTE: we use d = delta_x and multiply by 2*pi to get nm^-1
    qx_1d = np.fft.fftfreq(padded_shape[0], d=delta_x[0]) * 2 * np.pi
    qy_1d = np.fft.fftfreq(padded_shape[1], d=delta_x[1]) * 2 * np.pi
    qz_1d = np.fft.fftfreq(padded_shape[2], d=delta_x[2]) * 2 * np.pi

    qx_axis = np.fft.fftshift(qx_1d).astype(np.float32)
    qy_axis = np.fft.fftshift(qy_1d).astype(np.float32)
    qz_axis = np.fft.fftshift(qz_1d).astype(np.float32)
    
    #######################################################
    #                 PRECOMPUTE SECTION                  #
    # Lets precompute some stuff outside the frame loop   #
    #######################################################
    
    # precompute band-pass mask
    bp_mask = None
    if (args.qmin_filt is not None) or (args.qmax_filt is not None):
        if not args.quietly:
            print(f"Applying 3D band-pass: qmin_filt={args.qmin_filt}, qmax_filt={args.qmax_filt} (nm^-1)")
        bp_mask = band_pass_filter_q(
            shape=padded_shape,
            delta_x=delta_x,
            qmin=args.qmin_filt,
            qmax=args.qmax_filt
        )
    print('\nCalculating Structure Factor for Each Frame:')
    
    window = None
    if args.fft_window:
        '''
        NEW - compute fft window out of the loop (speed)
        '''
        shape = electron_density.shape
        window = (np.hamming(shape[0]).astype(np.float32)[:, None, None] *
                  np.hamming(shape[1]).astype(np.float32)[None, :, None] *
                  np.hamming(shape[2]).astype(np.float32)[None, None, :])
        
    n_frames = len(traj)
    
    kernel_cache = {}
    atom_kernels = []

    effective_voxel_size = float(np.mean(delta_x))

    for atom_radius, electrons in zip(atom_radii, atom_electrons):
        key = (round(atom_radius, 6), electrons)
        if key not in kernel_cache:
            kernel_cache[key] = make_gaussian_kernel(
                atom_radius,
                electrons,
                effective_voxel_size,
                args.n_sigma
            )
        atom_kernels.append(kernel_cache[key])

    # preallocate padded density and FFT output once
    # complex64 input preserves the same full FFT output shape as numpy_fft.fftn
    padded_shape_tuple = tuple(int(n) for n in padded_shape)

    density_padded = pyfftw.empty_aligned(padded_shape_tuple, dtype='complex64')
    ft_density = pyfftw.empty_aligned(padded_shape_tuple, dtype='complex64')

    density_padded[:] = 0.0
    ft_density[:] = 0.0

    fft_object = pyfftw.FFTW(
        density_padded,
        ft_density,
        axes=(0, 1, 2),
        direction='FFTW_FORWARD',
        flags=('FFTW_MEASURE',),
        threads=args.fft_threads
    )

    sf_accum_flat = sf_accum.ravel()
    ft_density_flat = ft_density.ravel()

    pad = args.padding
    inner = tuple(slice(pad, pad + n) for n in grid_shape)
    
    ##################################
    # Frame Loop code starts here... #
    ##################################
    
    for frame in tqdm(traj.xyz, total=n_frames, desc="frames", dynamic_ncols=True, position=0, leave=True):
        electron_density[:] = 0.0

        for atom_xyz, (g_width, gaussian_density) in zip(frame, atom_kernels):
            center = ((atom_xyz - global_min) // delta_x).astype(np.int32)
            update_electron_density(electron_density, center, gaussian_density, g_width, grid_shape)

        if args.fft_window:
            electron_density *= window
            
        density_padded[:] = 0.0
        density_padded[inner] = electron_density

        # run the pre-planned FFT; FFTW reads density_padded and writes in-place.
        fft_object()

        # accumulate via @njit
        accumulate_power_spectrum(ft_density_flat, sf_accum_flat)

    sf_accum /= float(n_frames)             # average over frames
    
    if bp_mask is not None:
        '''
        apply the bandpass filter out of the frame loop for peed
        '''
        sf_accum *= bp_mask.astype(np.float32)
    
    sf_accum = np.fft.fftshift(sf_accum)    # do the final fftshift just once, out of the loop (speed)
    
    return electron_density, sf_accum, (qx_axis, qy_axis, qz_axis) # q-axes are fftshifted and aligned with sf_accum

def interpolate_sf(sf_volume, q_axes):
    """
    interpolate the 3D Structure Factor within specified Q-space bounds.
    updated 2025_11_26
    
    inputs:
    sf_volume (an np.ndarray):   3D SF array [Nx, Ny, Nz]
    q_axes (a tuple):            (qx, qy, qz) 1D arrays

    returns:
    interpolated_SF_volume :     3D array on regular cube in Q
    q_interp_axes          :     np.array([qx_interp, qy_interp, qz_interp])
    """
    if not args.quietly:
        print('\nInterpolating 3D structure factor onto ' + str(args.points) + '^3 grid')

    qx_unique, qy_unique, qz_unique = q_axes

    qmax_available = min(
        np.max(np.abs(qx_unique)),
        np.max(np.abs(qy_unique)),
        np.max(np.abs(qz_unique))
    )

    qmax_use = args.qmax
    if qmax_use is None:
        qmax_use = qmax_available

    if qmax_use > qmax_available:
        warnings.warn(
            f"Requested interpolation qmax={qmax_use:.3f} nm^-1 exceeds "
            f"available FFT qmax={qmax_available:.3f} nm^-1. "
            f"Using {qmax_available:.3f} nm^-1 instead."
        )
        qmax_use = qmax_available

    qx_interp = np.linspace(-qmax_use, qmax_use, num=args.points)
    qy_interp = np.linspace(-qmax_use, qmax_use, num=args.points)
    qz_interp = np.linspace(-qmax_use, qmax_use, num=args.points)

    Qx_interp, Qy_interp, Qz_interp = np.meshgrid(qx_interp, qy_interp, qz_interp, indexing='ij')
    query_points = np.vstack((Qx_interp.ravel(), Qy_interp.ravel(), Qz_interp.ravel())).T

    interpolator = RegularGridInterpolator(
        (qx_unique, qy_unique, qz_unique),
        sf_volume,
        bounds_error=False,
        fill_value=0.0
    )

    interpolated_SF = interpolator(query_points)
    interpolated_SF_volume = interpolated_SF.reshape(Qx_interp.shape)

    return interpolated_SF_volume, np.array([qx_interp, qy_interp, qz_interp], dtype=object)


def _get_saxs_plane(sf_3d, q_axes, axis):
    '''
    sum a 3D structure factor along one axis and return the 2D SAXS image
    plus the two q-axes belonging to the displayed plane.
    '''
    qx_axis, qy_axis, qz_axis = q_axes
    saxs = np.sum(sf_3d, axis=axis)

    if axis == 0:      # sum x -> yz plane
        plane_name = "yz"
        q1_axis = qy_axis
        q2_axis = qz_axis
        
    elif axis == 1:    # sum y -> xz plane
        plane_name = "xz"
        q1_axis = qx_axis
        q2_axis = qz_axis
        
    elif axis == 2:    # sum z -> xy plane
        plane_name = "xy"
        q1_axis = qx_axis
        q2_axis = qy_axis
        
    else:
        raise ValueError("axis must be 0, 1 or 2")

    return plane_name, saxs.real, q1_axis, q2_axis


def _apply_symmetry(saxs, symmetric=False):
    '''
    mirror the 2D SAXS pattern.
    this just changes only the plotted image, not the underlying SF.
    '''
    if not symmetric:
        return saxs

    saxs_sym = saxs.copy()
    saxs_sym = saxs_sym + np.flipud(saxs_sym)
    saxs_sym = saxs_sym + np.fliplr(saxs_sym)

    return saxs_sym


def _crop_q_window(saxs, q1_axis, q2_axis, qmin=None, qmax=None):
    '''
    crop the 2D SAXS image using rectangular q-axis limits.
    '''
    q1_axis = np.asarray(q1_axis)
    q2_axis = np.asarray(q2_axis)

    if qmax is not None:
        q1_mask = np.abs(q1_axis) <= qmax
        q2_mask = np.abs(q2_axis) <= qmax
    else:
        q1_mask = np.ones_like(q1_axis, dtype=bool)
        q2_mask = np.ones_like(q2_axis, dtype=bool)

    if qmin is not None and qmin > 0.0:
        q1_mask &= np.abs(q1_axis) >= qmin
        q2_mask &= np.abs(q2_axis) >= qmin

    if not np.any(q1_mask) or not np.any(q2_mask):
        raise ValueError("q-window removed all data. Check qmin/qmax relative to q-axis range.")

    saxs_crop = saxs[np.ix_(q1_mask, q2_mask)]
    q1_crop = q1_axis[q1_mask]
    q2_crop = q2_axis[q2_mask]

    return saxs_crop, q1_crop, q2_crop


def _add_q_rings(ax, q1_axis, q2_axis, qstep=None, annotate=False):
    '''
    add circular q guide rings to an existing SAXS plot.
    q values are assumed to be in nm^-1.
    '''
    qmax_circle = min(abs(q1_axis[0]), abs(q1_axis[-1]),
                      abs(q2_axis[0]), abs(q2_axis[-1]))

    if qstep is None:
        qstep = qmax_circle / 5.0

    if qstep <= 0:
        raise ValueError("qstep must be positive")

    for q in np.arange(qstep, qmax_circle, qstep):
        circle = patches.Circle(
            (0.0, 0.0),
            q,
            color="white",
            fill=False,
            linestyle="--",
            zorder=5
        )
        ax.add_patch(circle)

        if annotate:
            ax.annotate(
                rf"${q:.2f} \, \mathrm{{nm}}^{{-1}}$",
                (0.0, -(q + qstep / 2.5)),
                color="white",
                fontsize=8,
                zorder=5
            )


def _plot_saxs_image(
    image,
    q1_axis,
    q2_axis,
    output_file,
    cmap="plasma",
    image_interp="None",
    lines=False,
    qstep=None,
    annotate=False,
    log_scale=False,
    eps=1e-20
):
    '''
    plot a single 2D SAXS image, either linear or log-scaled.
    '''
    if log_scale:
        plot_image = np.log(np.maximum(image, 0.0) + eps)
    else:
        plot_image = image

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        plot_image,
        cmap=cmap,
        extent=[q2_axis[0], q2_axis[-1], q1_axis[0], q1_axis[-1]],
        origin="lower",
        interpolation=image_interp
    )

    if lines:
        _add_q_rings(
            ax,
            q1_axis=q1_axis,
            q2_axis=q2_axis,
            qstep=qstep,
            annotate=annotate
        )
        ax.axis("off")
    else:
        ax.set_xlabel(r"$q_2$ / nm$^{-1}$")
        ax.set_ylabel(r"$q_1$ / nm$^{-1}$")

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_saxs_plot(args, structure_factor, q_interps):
    '''
    Make simulated SAXS plots from a 3D structure factor.
    '''
    sf_3d = structure_factor
    q_axes = tuple(np.asarray(q, dtype=float) for q in q_interps)

    for axis in range(3):
        plane_name, saxs, q1_axis, q2_axis = _get_saxs_plane(sf_3d, q_axes, axis)

        saxs = _apply_symmetry(saxs, symmetric=args.symmetric)
        saxs, q1_axis, q2_axis = _crop_q_window(saxs, q1_axis, q2_axis, qmin=args.qmin, qmax=args.qmax)

        np.savetxt(f"{args.output}_{plane_name}.csv", saxs.real, delimiter="," )

        _plot_saxs_image( image=saxs.real,
            q1_axis=q1_axis, q2_axis=q2_axis,
            output_file=f"{args.output}_{plane_name}.png",
            cmap=args.cmap, image_interp=args.image_interp,
            lines=args.lines, qstep=args.qstep,
            annotate=args.annotate, log_scale=False)

        _plot_saxs_image(
        image=saxs.real, q1_axis=q1_axis, q2_axis=q2_axis, output_file=f"{args.output}_{plane_name}_log.png",
        cmap=args.cmap, image_interp=args.image_interp, lines=args.lines,
        qstep=args.qstep, annotate=args.annotate, log_scale=True
        )

        if args.plot_integrated:
            print(f"Plotting integrated data for {plane_name}...")
            compute_radial_profile( saxs.real, q1_axis, q2_axis, ax_name=plane_name, output_basename=args.output, qmin=args.qmin, qmax=args.qmax, num_bins=360 )

    print("SAXS creation complete!")

def compute_radial_profile(image_data, qx_axis, qy_axis,
                           ax_name, output_basename,
                           qmin=None, qmax=None,
                           num_bins=360):

    '''
    get the radial integrated profile of some 2D SAXS data.

    image_data : 2D array
        - If log_scale=False, this should be *linear* intensity.
        - If log_scale=True, this should still be *linear* intensity;
          we will take the log at the end for plotting.
    qx(y)_axis:
    
    '''

    QX, QY = np.meshgrid(qx_axis, qy_axis, indexing='ij')
    q_mag = np.sqrt(QX**2 + QY**2)

    if qmin is None:
        qmin = np.min(q_mag)

    if qmax is None:
        qmax = np.max(q_mag)

    bin_edges = np.linspace(qmin, qmax, num_bins+1)
    q_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    integrated = np.zeros(num_bins)
    counts = np.zeros(num_bins)

    for i in range(num_bins):
        mask = (q_mag >= bin_edges[i]) & (q_mag < bin_edges[i+1])
        if np.any(mask):
            integrated[i] = np.sum(image_data[mask])
            counts[i] = np.sum(mask)

    good = counts > 0
    integrated[good] /= counts[good]

    # normal not log
    plt.figure()
    plt.plot(q_centers, integrated)
    plt.xlabel("Q (nm^-1)")
    plt.ylabel("Integrated Intensity")
    plt.savefig(f"{output_basename}_{ax_name}_integrated.png",
                bbox_inches='tight')

    # log
    plt.figure()
    plt.plot(q_centers, np.log(integrated + 1e-20))
    plt.xlabel("Q (nm^-1)")
    plt.ylabel("Log Integrated Intensity")
    plt.savefig(f"{output_basename}_{ax_name}_integrated_log.png",
                bbox_inches='tight')

# updated these two given we now save "less" in the 3D SF and stuff
def save_edensity_and_sf(electron_density, sf_volume, q_axes):
    if not args.quietly:
        print('Saving electron density as ' + args.output + '_edensity.npz')
    np.savez(args.output + '_edensity.npz', array=electron_density)

    if not args.quietly:
        print('Saving structure factor  as ' + args.output + '_sf.npz')
    qx, qy, qz = q_axes
    np.savez(args.output + '_sf.npz', sf=sf_volume, qx=qx, qy=qy, qz=qz)


def load_edensity_and_sf():
    print('Reloading edensity and SF from ' + args.reload)
    with np.load(args.reload + '_edensity.npz') as data:
        electron_density = data['array']

    with np.load(args.reload + '_sf.npz') as data:
        sf_volume = data['sf']
        qx = data['qx']
        qy = data['qy']
        qz = data['qz']

    return electron_density, sf_volume, (qx, qy, qz)

    
def initialize():
    parser = argparse.ArgumentParser(description='simusaxs')
    
    # trajectory options
    parser.add_argument('-top', '--topology', default='', type=str, help='input topology as an mdtraj readable format (e.g. .gro)')	
    parser.add_argument('-traj','--trajectory', default='', type=str, help='input trajectory as an mdtraj readable format (e.g. .trr, .xtc)')
    parser.add_argument('-rld','--reload', default='', type=str, help='reload edensity and sf from npz')
        
    # structure factor / electron density options
    parser.add_argument('-qmax','--qmax', default=25.0, type=float, help='Q_max for 3D structure factor in nm-1 (default = 2.0)')
    parser.add_argument('-qmin','--qmin', default=0.2, type=float, help='Q_min for 3D structure factor in nm-1 (default = 0.2)' )
    parser.add_argument('-vox','--voxel_size', default=0.05, type=float, help='real space (nm) voxel size for the electron density grid (default = 0.05)')
    parser.add_argument('-pad','--padding', default=20, type=int, help='zero-padding size for 3D structure factor before fft (int; default = 20)')
    parser.add_argument('-win','--fft_window', action='store_false', help='Turn off FFT-window filtering')
    parser.add_argument('-sig','--n_sigma', default=3, type=int, help='consider atomic electron density to n_sigma * sigma in the Gaussian term (default = 3)')
    parser.add_argument('-iim', '--image_interp', default='None', type=str, help='Set image interpolation method (default = None)')
    
    # interpolation control    
    parser.add_argument('-pts','--points', default=100, type=int, help='Number of interpolation points along each Q-axis (default = 100)')
    parser.add_argument('-int', '--interpolation', action='store_true', help='Turn on 3D SF interpolation')
    
    # control FFT windowing/filtering
    parser.add_argument('-qmin_filt', default=None, type=float, help='Q_min for 3D SF band-pass (nm-1, default=None = low angle-Q cut)')
    parser.add_argument('-qmax_filt', default=None, type=float, help='Q_max for 3D SF band-pass (nm-1, default=None = no high-Q cut)')
    
    # output control (files)
    parser.add_argument('-q', '--quietly', action='store_true', help='disable writing most output to terminal.')
    parser.add_argument('-o','--output', default='simusaxs_', type=str, help='output file basename')
    parser.add_argument('-save','--save_npz', action='store_true', help='enable saving edensity map and 3D-SF to .npz files.')
    
    #control plotting options:
    parser.add_argument('-lines', '--lines', action='store_true', help='put lines on plot (default: False)')
    parser.add_argument('-annotate', '--annotate', action='store_false', help='put annotations on lines on plot (default: False)')
    parser.add_argument('-qstep','--qstep', default=None, type=float, help='step-size in Q for lines')
    parser.add_argument('-cmap','--cmap', default='plasma', type=str, help='colormap of plot')
    parser.add_argument('-sym','--symmetric', action='store_true', help='make plot (not data!) symmetric by mirroring L/R and U/D')
    parser.add_argument('-plt','--plot_integrated', action='store_true', help='Turn on radial integration; produce plots (useful for comparison w/ experimental data)')
     
    #trajectory frame options	
    parser.add_argument('-b','--begin', default='0', type=int, help='frame no. of trajectory to start at (default = 0; i.e. frame #1)')
    parser.add_argument('-s','--step', default=None, type=int, help='step size for frame in trajectory to start at (default = None, i.e. all frames)')
    parser.add_argument('-e','--end', default='-1', type=int, help='frame no. of trajectory to end at (default = -1; i.e. all)')
    
    #new (09-06-2026) multithreaded FFT options
    parser.add_argument('-nt', '--fft_threads', default=1, type=int, help='Number of threads for pyFFTW FFTs')
    return parser

if __name__ == "__main__":


    args = initialize().parse_args()
    
    if args.voxel_size <= 0:
        raise ValueError("--voxel_size must be positive")
    
    # specific pyfftw thread configs
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
    pyfftw.config.NUM_THREADS = args.fft_threads

    if args.quietly == False:
        #pointless ascii text
        print('\n')
        
        print(r"  ______   _____  ____    ____  _____  _____   ______        _       ____  ____   ______     ")  
        print(r".' ____ \ |_   _||_   \  /   _||_   _||_   _|.' ____ \      / \     |_  _||_  _|.' ____ \    ")
        print(r"| (___ \_|  | |    |   \/   |    | |    | |  | (___ \_|    / _ \      \ \  / /  | (___ \_|   ")
        print(r" _.____`.   | |    | |\  /| |    | '    ' |   _.____`.    / ___ \      > `' <    _.____`.    ")
        print(r"| \____) | _| |_  _| |_\/_| |_    \ \__/ /   | \____) | _/ /   \ \_  _/ /'`\ \_ | \____) |   ")
        print(r" \______.'|_____||_____||_____|    `.__.'     \______.'|____| |____||____||____| \______.'   ")
        
        print('\nVersion 0.3 - Richard Mandle, UoL, 2023-2026')
                                                                                       
        _ = get_platform()
        print('\n')
    if args.quietly == True:
        print('Simusaxs is running quietly')
    
    if args.reload == '':
        traj = load_and_process_traj(traj=args.trajectory, top=args.topology)
        electron_density, sf_volume, q_axes = compute_3d_sf(traj[args.begin:args.end:args.step])
        if args.save_npz:
            save_edensity_and_sf(electron_density, sf_volume, q_axes)
    else:
        electron_density, sf_volume, q_axes = load_edensity_and_sf()

    if args.reload == '' and (args.trajectory == '' or args.topology == ''):
        print('No trajectory/topology supplied, no data supplied to reload;\n please provide with -traj/-top or -rld flags!')
        sys.exit()

    del electron_density # clear it from RAM
    
    print('SF shape: ' + str(np.shape(sf_volume)))

    if args.interpolation:
        interpolated_SF_volume, q_array = interpolate_sf(sf_volume, q_axes)
        make_saxs_plot(args, interpolated_SF_volume, q_array)
        print('iSF shape: ' + str(np.shape(interpolated_SF_volume)))
    else:
        make_saxs_plot(args, sf_volume, np.array(q_axes, dtype=object))
        
    
