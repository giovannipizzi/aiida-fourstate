#!/usr/bin/env runaiida
"""Launch script for the MagneticExchangeWorkChain."""
import copy
import numpy as np

from aiida import orm
from aiida.engine import submit, calcfunction
from aiida_common_workflows.common import ElectronicType
import ase.io
from ase.geometry import find_mic

from aiida_fourstate import MagneticExchangeWorkChain

GROUP_NAME = 'FourStateProtocol-convergence/CrI3/LDA/abinit/1x1x1cell'

## THOR
code_label = 'abinit-10.0.3@thor-micromamba'
num_machines = 1
num_mpiprocs_per_machine = 48


protocol = 'custom'

kpoints = [12, 12, 1] # For a unit cell, I do a denser k-point mesh
# Will be stored later
tsmear = 7.34986e-05 / 2 # smearing: 1 meV (written in Hartree)

# LDA four-state v0 protocol for Abinit (with coarser k-points)
custom_protocol_BASE = {
    'base': {'abinit': {'parameters': {
        'autoparal': 1,
        'chkprim': 0,
        'chksymbreak': 0,
        'dilatmx': 1.0,
        'ecutsm': 0.0,
        'fband': 2.0,
        'ionmov': 22,
        'nstep': 300,
        'occopt': 3,
        'optcell': 0,
        'shiftk': [[0.0, 0.0, 0.0]],
        'tolvrs': 1e-10,
        #'npfft': 1, # no MPI-FFT, sometimes it complains
        #'iscf': 2, # Slower but safer, in 2x2x1 wasn't converging        
        'tsmear': tsmear}},
    'kpoints': kpoints},
 'cutoff_stringency': 'normal',
 'description': 'Protocol for the 4-state verification with LDA functional and PseudoDojo pseudopotentials.',
 'name': 'fourstate-LDA-v1',
 'pseudo_family': 'PseudoDojo/0.4/LDA/SR/standard/psp8'}


# To decide if we want to track also provenance here with a calcfunction
@calcfunction
def create_supercell(structure, supercell_matrix):
    """Create a supercell from the input structure.
    
    :param structure: AiiDA StructureData
    :param supercell_matrix: List or tuple of 3 integers [na, nb, nc]
    :return: AiiDA StructureData for the supercell
    """
    ase_structure = structure.get_ase()
    supercell_ase = ase_structure * supercell_matrix
    
    return orm.StructureData(ase=supercell_ase)

def group_by_distance(dist_list, threshold=0.1):
    """
    dist_list: list of tuples (idx, symbol, distance)
    threshold: maximum spread within one group (angstrom)

    Returns: 
        [
            {
                'average_dist': float,
                'items': [(idx, distance), ...]  # sorted by idx
            },
            ...
        ]
    ]
    """
    if not dist_list:
        return []

    # Sort items by distance first
    dist_list = sorted(dist_list, key=lambda x: x[1])

    groups = []
    current_group = [dist_list[0]]

    for item in dist_list[1:]:
        if abs(item[1] - current_group[0][1]) <= threshold:
            # same group
            current_group.append(item)
        else:
            # close previous group
            groups.append(current_group)
            current_group = [item]

    # append final group
    groups.append(current_group)

    # Convert groups to list of dictionaries
    result = []
    for g in groups:
        avg = sum(x[1] for x in g) / len(g)
        g_sorted = sorted(g, key=lambda x: x[0])  # sort by idx
        result.append({
            'average_dist': avg,
            'items': g_sorted
        })

    # Sort outer list by average distance
    result = sorted(result, key=lambda x: x['average_dist'])

    return result

def find_neighbor(site1, neigh_idx, filter_element, atoms):
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    r0 = positions[site1]

    dist_list = []
    for i, r in enumerate(positions):
        if i == site1:
            continue

        if atoms[i].symbol != filter_element:
            continue

        # Minimum-image displacement vector
        dr = r - r0
        dr_mic = find_mic([dr], cell, pbc)[0]

        # Distance
        dist = np.linalg.norm(dr_mic)

        dist_list.append((i, dist))

    # Sort by distance, then by index
    dist_list = sorted(dist_list, key=lambda x: (x[1], x[0]))
    dist_grouped_by_distance = group_by_distance(dist_list, threshold=0.1)

    assert neigh_idx >= 1
    assert neigh_idx <= len(dist_grouped_by_distance)
    group = dist_grouped_by_distance[neigh_idx-1]
    return group['items'][0]


def launch_magnetic_exchange_calculation(neigh_idx = 1, cutoff_wfc=None, ask_for_confirmation=True):
    """Launch a magnetic exchange coupling calculation.
    
    :param neigh_idx: Index of the neighbor to consider (1 means first neighbor)
    :param cutoff_wfc: Kinetic energy cutoff for wavefunctions (in Ry). If None, use default from protocol.
    :param ask_for_confirmation: If True, ask for user confirmation before submitting.
    """
    global code_label, num_mpiprocs_per_machine, kpt_node, custom_protocol

    CUTOFF_DUAL = 4. # Hardcoded for now since I know I'm using norm-conserving pseudopotentials

    supercell_matrix = [1, 1, 1] ## For convergence I do the 1x1x1 cell
    site1 = 0 # Cr

    ase_atoms = ase.io.read('../cri3_primitive.cif')
    structure_unitcell = orm.StructureData(ase=ase_atoms)

    if supercell_matrix == [1, 1, 1]:
        structure_supercell = structure_unitcell
    else:
        # Create the supercell structure
        structure_unitcell.store()
        structure_supercell = create_supercell(structure_unitcell, supercell_matrix)
    
    print(f"Unit cell has {len(structure_unitcell.sites)} atoms")
    print(f"Supercell has {len(structure_supercell.sites)} atoms")
    
    site2, dist = find_neighbor(site1 = site1, neigh_idx=neigh_idx, filter_element="Cr", atoms=structure_supercell.get_ase())

    magnetization_per_site = [3. if structure_supercell.get_kind(kind_name).symbol == "Cr" else 0. for kind_name in structure_supercell.get_site_kindnames()]
    magnetization_magnitude = 3.0 # For the two sites to flip

    ######## ADAPT PROTOCOL WITH CUTOFFS, IF ASKED ########
    custom_protocol = copy.deepcopy(custom_protocol_BASE)

    if cutoff_wfc is not None:            
        custom_protocol['base']['abinit']['parameters']['ecut'] = cutoff_wfc / 2.
    #######################################################

    # Generator inputs for the common workflows
    generator_inputs = {
        'engines': {
            'relax': {
                'code': code_label,
                'options': {
                    'resources': {
                        'num_machines': num_machines,
                        'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
                    },
                    'max_wallclock_seconds': 12*3600, # 12 hours
                },
            }
        },
        'protocol': protocol,
        'custom_protocol': custom_protocol if protocol == 'custom' else None,
        'electronic_type': ElectronicType.METAL.value, 
        'magnetization_per_site': magnetization_per_site
    }
        
    print(f"\nMagnetic sites in supercell:")
    print(f'{site1=}, {site2=}')
    atom1 = structure_supercell.get_ase()[site1]
    atom2 = structure_supercell.get_ase()[site2]
    print(f"  Site 1: {atom1.symbol}, index {site1}")
    print(f"Looking for {neigh_idx}-th neighbor of site {site1}")
    print(f"  Site 2: {atom2.symbol}, index {site2}")
    vector = atom2.position - atom1.position
    print(f"  Distance between selected sites: {np.linalg.norm(vector):.2f} ang\n")

    inputs = {
        'structure': structure_supercell,
        'site1': orm.Int(site1),
        'site2': orm.Int(site2),
        'magnetization_magnitude': orm.Float(magnetization_magnitude),
        'generator_inputs': generator_inputs,
        'engine_name': orm.Str('abinit'),
    }

    if ask_for_confirmation:
        print(f"Submitting J({neigh_idx}) for CrI3 to {code_label} on {num_machines} node(s) with {num_mpiprocs_per_machine=}.")
        print("Submit? [Ctrl+C to stop]")
        input()

    print("Submitting (ABINIT) MagneticExchangeWorkChain...")
    node = submit(MagneticExchangeWorkChain, **inputs)
    print(f"\nSubmitted: J({neigh_idx}), PK = {node.pk}")
    node.label = f"CrI3 J({neigh_idx}) calculation"
    node.base.extras.set('magnetic_exchange_neigh_idx', neigh_idx)
    node.base.extras.set('cutoff_wfc', cutoff_wfc)

    group, created = orm.Group.collection.get_or_create(GROUP_NAME)
    group.add_nodes([node])

    return node
    

if __name__ == '__main__':
    for neigh_idx in [1]:
        # Cutoff from PseudoDojo: 94 Ry for Cr
        for cutoff_wfc in range(50, 111, 10):  # from 50 to 110 Ry, step 10 Ry
            print(f"\n\nLaunching QE calculation of J({neigh_idx})*S^2 (ecutwfc={cutoff_wfc})\n")
            launch_magnetic_exchange_calculation(neigh_idx=neigh_idx, cutoff_wfc=cutoff_wfc, ask_for_confirmation=False)
    print()
    print(f"\nAll calculations submitted and added to group {GROUP_NAME}.")
