#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numba
from numba import njit, prange
import psutil # used when we assign resolution as 'max'

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
    rotation matrix required to transform the orientation of
    vec1 into vec2. 
    
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
    radii_dict = {}
    try:
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip().split('/')
                atomic_number = int(line[0].strip())
                atomic_symbol = line[1].strip()
                atomic_radius = int(line[2].strip())
                radii_dict[atomic_symbol] = {'element': atomic_symbol, 'number': atomic_number, 'radius': atomic_radius}
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {str(e)}")
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
    return system_info
    
def determine_max_res():
    '''
    little function that works out the maximum resolution we can have for our electron density grid by
    determining how much free memory we have.
    
    STATUS - been tested on WIN10; does it work the same on the HPC? 
    '''
    
    free_vmem = psutil.virtual_memory().available
    bytes_per_element = np.dtype(np.float32).itemsize
    
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

def get_order_parameter(traj):
    '''
    Calculate the nematic order paramter in traj
    
    doesn't seem to be used?
    '''
    mol_indices = [[n+x for x in range(int(traj.n_atoms/traj.n_residues))] for n in range(0, traj.n_atoms, int(traj.n_atoms/traj.n_residues))]
    P2 = md.compute_nematic_order(traj, mol_indices)
    if args.quietly == False:
        print(P2)
    return P2

def get_traj_properties(traj):
    '''
    Return some properties of the trajectory
    '''
    properties = {
        'mols': traj.n_residues,
        'atoms': traj.n_atoms,
        'frames': traj.n_frames,
        'timestep': traj.time,
        'size': np.mean(traj.unitcell_lengths, axis=0)
    }
    if args.quietly == False:
        print('\n')
        print('-'*50)
        print('Trajectory Information:')
        print(f"{properties['atoms']} atoms in {properties['mols']} molecules/residues")
        print(f"{properties['frames']} frames separated by {properties['timestep'][1]} picoseconds, total run time of {(properties['timestep'][1]*properties['frames'])/1000} nanoseconds")
        print(f"Unit cell dimensions of X = {properties['size'][0]}nm, Y = {properties['size'][1]}nm, Z = {properties['size'][2]}nm")
        print('-'*50)
    return properties
    
def compute_rescaling_factors(traj):
    '''
    Compute rescaling factors based on the maximum dimension of the unit cell.
    '''
    max_lengths = np.array([np.max(traj.unitcell_lengths[:,0]),
                   np.max(traj.unitcell_lengths[:,1]),
                   np.max(traj.unitcell_lengths[:,2])])
    
    rescaling_factors = max_lengths / traj.unitcell_lengths
    dimensions = ['X-', 'Y-', 'Z-']

    if args.quietly == False:
        print('-'*50)
        for x, y in zip(np.mean(rescaling_factors, axis=0), dimensions):
            print(f'variance in {y} dim = {np.round((x - 1) * 100, 3)} %')
        print('-'*50)
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

def centre_trajectory(traj):
    """
    simply centres a trajectory
    """

    new_traj = traj.center_coordinates(traj)
        
    return new_traj

def prepare_trajectory(traj,new_orientation=np.array([0,1,0])):
    '''
    Prepares a trajectory for SAXS computation by performing multiple functions.
    A wrapper, basically
    '''
    if args.quietly == False:
        print('-'*50)
        print('preparing trajectory for SAXS computation')
    
    traj = apply_rescaling(traj, compute_rescaling_factors(traj))
    rotate_trajectory(traj, rotation_matrix_from_vectors(np.mean(get_director(traj),axis=0), new_orientation))
    centre_trajectory(traj)
    
    if args.quietly == False:
        print('completed trajectory preparation')
        print('-'*50)
    
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

def compute_structure_factor(electron_density):
    """
    Compute the 3D Structure Factor from the electron density.
    
    Args:
        electron_density (np.ndarray): A 3D array representing the electron density.
 
    Returns:
        np.ndarray: A 3D array representing the Structure Factor.
    """

    structure_factor = np.fft.fftn(electron_density) # do 3d fft
    structure_factor = np.fft.fftshift(structure_factor) # Shift the Zero Frequency Component to the Center
    structure_factor = np.real(structure_factor * structure_factor.conjugate())
    
    return structure_factor

def compute_scattering_vector(electron_density, resolution):
    shape = electron_density.shape
    qx = np.fft.fftfreq(shape[0], d=resolution) * 2 * np.pi
    qy = np.fft.fftfreq(shape[1], d=resolution) * 2 * np.pi
    qz = np.fft.fftfreq(shape[2], d=resolution) * 2 * np.pi
    
    qx, qy, qz = np.meshgrid(qx, qy, qz, indexing='ij')
    
    return qx, qy, qz


def band_pass_filter(shape, q_min, q_max, delta_x):
    '''
    used to mask out features outside of our q-range 
    use in the frequency domain
    '''
    x, y, z = np.meshgrid(np.fft.fftfreq(shape[0]),
                          np.fft.fftfreq(shape[1]),
                          np.fft.fftfreq(shape[2]), indexing='ij')

    
    # Distance from the origin (0,0,0) in reciprocal space
    q = np.sqrt((x**2 / delta_x[0]**2) + (y**2 / delta_x[1]**2) + (z**2 / delta_x[2]**2))
    
    # Create a binary mask
    mask = np.logical_and(q >= q_min, q <= q_max)
    
    return mask

def compute_3d_sf(traj):
    '''
    Computes the 3D electron density and 3D strcture factor from a traj object
    
    args:
    traj - MD traj trajectory object
    q_min - minimum scattering vector to probe
    q_max - maximum scattering vector to probe
    resolution - the extent of each voxel in Q-space (e.g. 2pi / d)
    padding - size of array zero-padding to use in 3dFFT
    fft_window - use a windowing function to improve the fft
    n_sigma - consider electron density to n_sigma * sigma in the Gaussian term
    '''
    
    if args.n_sigma < 2:
        print('Adjusting Gaussian sigma value to 2 (you specified ' + str(args.n_sigma) + ', which would return an error)')
        args.n_sigma = 2
    _, atom_radii, _, atom_electrons = get_atom_information(traj)
    
    global_min = np.amin(traj.xyz, axis=(0, 1))
    global_max = np.amax(traj.xyz, axis=(0, 1))
    
    simulation_dimensions = global_max - global_min

    grid_shape = np.ceil((simulation_dimensions) / args.resolution).astype(int) # calculate grid on the fly

    delta_x = simulation_dimensions / grid_shape  # Calculate the actual grid spacing in reciprocal space
    
    available_vmem = psutil.virtual_memory().available / (1024**3)
    required_vmem = (2*(np.dtype(np.float32).itemsize * np.prod(grid_shape, dtype=np.int64)) / (1024**3)) + (np.dtype(np.complex128).itemsize * np.prod(grid_shape, dtype=np.int64)) / (1024**3) # int64 so we don't overflow
    
    if args.quietly == False:
        print('\n')
        print('-'*50)
        print('\nSimulation Dimensions: ' + str(simulation_dimensions) + ' nm')
        print('Grid Shape: ' + str(grid_shape)) 
        print('Resolution ' + str(delta_x) + ' nm') # should be close to specified resolution
        print(f"Q(min) from data: {2 * np.pi / np.max(simulation_dimensions):.3f} nm-1")
        print(f"Q(max) from data: {2 * np.pi / np.max(delta_x):.3f} nm-1")
        print(f"Q(min) requested: {args.qmin:.3f} nm-1")
        print(f"Q(max) requested: {args.qmax:.3f} nm-1")
        print('\n')
        print(f"total RAM available: {available_vmem:.3f} GB")
        print(f"RAM required for edens: {(np.dtype(np.float32).itemsize * np.prod(grid_shape, dtype=np.int64)) / (1024**3):.3f} GB")
        print(f"RAM required for 3D-SF: {(np.dtype(np.float32).itemsize * np.prod(grid_shape, dtype=np.int64)) / (1024**3):.3f} GB")
        print(f"RAM required for complex 3D SF: {(np.dtype(np.complex128).itemsize * np.prod(grid_shape, dtype=np.int64)) / (1024**3):.3f} GB")
        print(f"total RAM required : {required_vmem :.3f} GB") # estimate RAM needed for an array;
        if (required_vmem) < available_vmem:
            print('   !!! Enough RAM is available to permit higher resolution !!!')
            memory_ratio = available_vmem / (required_vmem)
            potential_increase = np.ceil(memory_ratio ** 1/3)
            print(f"   You could potentially increase the resolution by a factor of {potential_increase:.2f}")
            print(f"   This is estimated to using -r {args.resolution / potential_increase}.")
        print('-'*50)
        print('\n')   
        
    electron_density = np.zeros(shape=grid_shape,dtype=np.float32)
   
    padded_shape = np.array(grid_shape) + 2 * args.padding  # Calculate bp_filter BEFORE the loop!!!
    bp_filter = band_pass_filter(padded_shape, args.qmin, args.qmax, delta_x)

    structure_factor = np.zeros(shape=(*padded_shape, 4), dtype=np.float32) #array to store qx, qy, qz, and SF
    
    qx, qy, qz = np.meshgrid(np.fft.fftfreq(padded_shape[0]),     # Generate Q-space coordinates
                             np.fft.fftfreq(padded_shape[1]),
                             np.fft.fftfreq(padded_shape[2]), indexing='ij')
    
    print('\nCalculating Structure Factor for Each Frame:')
    for frame in tqdm(traj.xyz, total=len(traj), desc="frames", dynamic_ncols=True, position=0, leave=True):
        electron_density[:] = 0  # set to zero w/out re-allocating RAM
        for atom_xyz, atom_radius, electrons in zip(frame, atom_radii, atom_electrons):
            
            g_width = int(args.n_sigma * atom_radius / args.resolution) # cover n-sigma (atom_radius / resolution) from centre of atom
            center = ((atom_xyz - global_min) // delta_x).astype(int)
            grid_indices = create_indices(g_width)
            distances_sq = sum_squared_distances(grid_indices, g_width)
            gaussian_density = electrons / ((atom_radius / args.resolution) ** 3) * np.exp(-0.5 * (distances_sq / (atom_radius / args.resolution)**2))
            update_electron_density(electron_density, center, gaussian_density, g_width, grid_shape) # get dens from jitted function
            
        if args.fft_window == True:
            shape = electron_density.shape
            window = np.hamming(shape[0])[:, None, None] * np.hamming(shape[1])[None, :, None] * np.hamming(shape[2])[None, None, :]
            electron_density *= window
 
        ft_density = np.fft.fftn(np.pad(electron_density, args.padding, mode='constant', constant_values=0)) * bp_filter
        
        sf = np.real(np.fft.fftshift(ft_density) * np.conjugate(np.fft.fftshift(ft_density)))
        
        structure_factor[:, :, :, 0] = qx
        structure_factor[:, :, :, 1] = qy
        structure_factor[:, :, :, 2] = qz
        structure_factor[:, :, :, 3] += sf  # Accumulate SF values across frames

    return electron_density, structure_factor

def interpolate_sf(structure_factor):
    """
    Interpolate the 3D Structure Factor within specified Q-space bounds.
    
    Args:
    structure_factor (numpy.ndarray): 4D array containing Qx, Qy, Qz, and SF.
    points (int, optional): Number of interpolation points along each Q-axis. Default is 100.
    q_max (float, optional): Maximum absolute value for Qx, Qy, and Qz. Default is 0.25 (nm-1)!
        
    Returns:
    3D array of interpolated SF values within the specified Q-space bounds.
    """
    if args.quietly == False:
        print('\nInterpolating 3D structure factor onto ' + str(args.points) + '^3 grid')

    qx_unique = np.unique(structure_factor[:, :, :, 0])    # Get unique grid points for each Q-dimension
    qy_unique = np.unique(structure_factor[:, :, :, 1])
    qz_unique = np.unique(structure_factor[:, :, :, 2])

    qx_interp = np.linspace(-args.qmax, args.qmax, num=args.points)  # Adjust num as needed
    qy_interp = np.linspace(-args.qmax, args.qmax, num=args.points)
    qz_interp = np.linspace(-args.qmax, args.qmax, num=args.points)

    Qx_interp, Qy_interp, Qz_interp = np.meshgrid(qx_interp, qy_interp, qz_interp, indexing='ij')

    query_points = np.vstack((Qx_interp.ravel(), Qy_interp.ravel(), Qz_interp.ravel())).T

    interpolator = RegularGridInterpolator((qx_unique, qy_unique, qz_unique), structure_factor[:, :, :, 3], bounds_error=False, fill_value=None)

    interpolated_SF = interpolator(query_points)

    interpolated_SF_volume = interpolated_SF.reshape(Qx_interp.shape)

    return interpolated_SF_volume, np.array([qx_interp, qy_interp, qz_interp])

def make_saxs_plot(structure_factor, q_interps = None): # q_step = 0.05, cmap='magma', lines=True):
    axis_to_name = {0: 'yz', 1: 'xz', 2: 'xy'}
    
    if q_interps == None:
        qx_unique = np.unique(structure_factor[:, :, :, 0])    # Get unique grid points for each Q-dimension
        qy_unique = np.unique(structure_factor[:, :, :, 1])
        qz_unique = np.unique(structure_factor[:, :, :, 2])

        qx_interp = np.linspace(-args.qmax, args.qmax, num=args.points)  # Adjust num as needed
        qy_interp = np.linspace(-args.qmax, args.qmax, num=args.points)
        qz_interp = np.linspace(-args.qmax, args.qmax, num=args.points)

        Qx_interp, Qy_interp, Qz_interp = np.meshgrid(qx_interp, qy_interp, qz_interp, indexing='ij')
        q_interps = np.array([qx_interp, qy_interp, qz_interp])

        structure_factor = structure_factor[:,:,:,3]
        
    for axis in range(3):
        saxs = np.sum(structure_factor, axis=axis)
        
        if args.symmetric == True: # make it symmetric by adding the left/right and top/bottom hemispheres.
            saxs = saxs + np.flipud(saxs)
            saxs = saxs + np.fliplr(saxs) 
            
        saxs = saxs[~np.all(saxs == 0, axis=1)][:, ~np.all(saxs == 0, axis=0)] # cut out zero rows/columns
        if args.log == True:
            saxs = np.log(saxs.real)
            
        fig, ax = plt.subplots(figsize=(8, 8))
        if not args.lines:
            ax.imshow(saxs.real, cmap=args.cmap)
            
        if args.lines:
            q_interp_x, q_interp_y = q_interps[(axis + 1) % 3], q_interps[(axis + 2) % 3]
            
            if args.qstep == None:
                args.qstep = (min(q_interp_x[-1], q_interp_y[-1])) / 5
        
            center_x = (q_interp_x[0] + q_interp_x[-1]) / 2 # find the centre in x (and y)
            center_y = (q_interp_y[0] + q_interp_y[-1]) / 2
            
            # This seems to give very odd numbers??
            q_pixel_size_x = (q_interp_x[-1] - q_interp_x[0]) / saxs.shape[1] # determine pixel size in Q
            q_pixel_size_y = (q_interp_y[-1] - q_interp_y[0]) / saxs.shape[0]
            
            q_pixel_ratio = np.mean([q_pixel_size_x, q_pixel_size_y])
            
            ax.imshow(saxs.real, cmap=args.cmap, extent=[q_interp_x[0], q_interp_x[-1], q_interp_y[0], q_interp_y[-1]], interpolation=args.image_interp)
            
            for q in np.arange(args.qstep, min(q_interp_x[-1], q_interp_y[-1]), args.qstep): # This loop draws circles at each q_step
                circle = patches.Circle((center_x, center_y), q, color='white', fill=False, linestyle='--', zorder=5)
                ax.annotate(rf"${q:.2f} \, \mathrm{{nm}}^{{-1}}$", (center_x, center_y - (q + args.qstep/2.5)), color='white', fontsize=8, zorder=5)
                ax.add_patch(circle)
                ax.axis('off')
        plt.savefig(f'{args.output}_{axis_to_name[axis]}'+ ('_log'*args.log) + '.png', bbox_inches='tight')
        plt.close(fig)
    print('SAXS creation complete!')
    
def save_edensity_and_sf(electron_density, structure_factor):
    if args.quietly == False:
        ('Saving electron density as ' + args.output + '_edensity.npz')
    np.savez(args.output + '_edensity.npz', array = electron_density)
    
    if args.quietly == False:
        ('Saving structure factor  as ' + args.output + '_sf.npz')
    np.savez(args.output + '_sf.npz', array = structure_factor)
    
def load_edensity_and_sf():
    print('Reloading edensity and SF from ' + args.reload)
    with np.load(args.reload + '_edensity.npz') as data:
        electron_density = data['array']

    with np.load(args.reload + '_sf.npz') as data:
        structure_factor = data['array']
    
    return electron_density, structure_factor
    
def initialize():
    parser = argparse.ArgumentParser(description='simusaxs')
    
    # trajectory options
    parser.add_argument('-top', '--topology', default='', type=str, help='input topology as an mdtraj readable format (e.g. .gro)')	
    parser.add_argument('-traj','--trajectory', default='', type=str, help='input trajectory as an mdtraj readable format (e.g. .trr, .xtc)')
    parser.add_argument('-rld','--reload', default='', type=str, help='reload edensity and sf from npz')
        
    # structure factor / electron density options
    parser.add_argument('-qmax','--qmax', default=2.0, type=float, help='Q_max for 3D structure factor in nm-1')
    parser.add_argument('-qmin','--qmin', default=0.2, type=float, help='Q_min for 3D structure factor in nm-1')
    parser.add_argument('-r','--resolution', default=0.05, type=float, help='resolution for 3D structure factor in nm-1')
    parser.add_argument('-pad','--padding', default=20, type=int, help='zero-padding size for 3D structure factor before fft (int)')
    parser.add_argument('-win','--fft_window', action='store_false', help='Turn off FFT-window filtering')
    parser.add_argument('-sig','--n_sigma', default=3, type=int, help='consider atomic electron density to n_sigma * sigma in the Gaussian term ')
    parser.add_argument('-iim', '--image_interp', default='None', type=str, help='Set image interpolation method')
    
    # interpolation control    
    parser.add_argument('-pts','--points', default=100, type=int, help='Number of interpolation points along each Q-axis. Default is 100.')
    parser.add_argument('-int', '--interpolation', action='store_true', help='Turn on 3D SF interpolation')
    
    # output control (files)
    parser.add_argument('-q', '--quietly', action='store_true', help='disable writing most output to terminal.')
    parser.add_argument('-o','--output', default='simusaxs_', type=str, help='output file basename')
    parser.add_argument('-save','--save_npz', action='store_true', help='disable saving edensity map and 3D-SF to .npz files.')
    
    #control plotting options:
    parser.add_argument('-l','--lines', default=True, type=bool, help='put lines on plot (True/False)')
    parser.add_argument('-stp','--qstep', default=None, type=float, help='step-size in Q for lines')
    parser.add_argument('-cmap','--cmap', default='plasma', type=str, help='colormap of plot')
    parser.add_argument('-sym','--symmetric', action='store_false', help='make plot (not data!) symmetric by mirroring L/R and U/D')
    parser.add_argument('-log','--log', action='store_true', help='plot using log scale')
 
    #trajectory frame options	
    parser.add_argument('-b','--begin', default='0', type=int, help='frame no. of trajectory to start at')
    parser.add_argument('-s','--step', default=None, type=int, help='step size for frame in trajectory to start at')
    parser.add_argument('-e','--end', default='-1', type=int, help='frame no. of trajectory to end at')
    return parser

if __name__ == "__main__":

    args = initialize().parse_args()
    
    if args.quietly == False:
        #pointless ascii text
        print('\n')
        print("  ______   _____  ____    ____  _____  _____   ______        _       ____  ____   ______   ")  
        print(".' ____ \ |_   _||_   \  /   _||_   _||_   _|.' ____ \      / \     |_  _||_  _|.' ____ \  ")
        print("| (___ \_|  | |    |   \/   |    | |    | |  | (___ \_|    / _ \      \ \  / /  | (___ \_| ")
        print(" _.____`.   | |    | |\  /| |    | '    ' |   _.____`.    / ___ \      > `' <    _.____`.  ")
        print("| \____) | _| |_  _| |_\/_| |_    \ \__/ /   | \____) | _/ /   \ \_  _/ /'`\ \_ | \____) | ")
        print(" \______.'|_____||_____||_____|    `.__.'     \______.'|____| |____||____||____| \______.' ")
        print('Version 0.1 - R.J.M, UoL, 2023')                                                                                           
        print('\n')
    if args.quietly == True:
        print('Running quietly')
    
    if args.reload == '':
        traj = load_and_process_traj(traj = args.trajectory, top = args.topology)
        
        electron_density, structure_factor = compute_3d_sf(traj[args.begin:args.end:args.step])
        if args.save_npz == True:
            save_edensity_and_sf(electron_density, structure_factor)
    
    if args.reload != '':
            electron_density, structure_factor = load_edensity_and_sf()
    
    if args.reload == '' and (args.trajectory == '' or args.topology == ''):
        print('No trajectory/topology supplied, no data supplied to reload;\n please provide with -traj/-top or -rld flags!')
        sys.exit() 
        
    del electron_density # clear it from RAM
    
    print('SF shape: ' + str(np.shape(structure_factor[:,:,:,3])))
    
    if args.interpolation == True:
        interpolated_SF_volume, q_array = interpolate_sf(structure_factor)
        make_saxs_plot(interpolated_SF_volume, q_array)
        print('iSF shape: ' + str(np.shape(interpolated_SF_volume)))
    
    if args.interpolation == False:
        make_saxs_plot(structure_factor)
   