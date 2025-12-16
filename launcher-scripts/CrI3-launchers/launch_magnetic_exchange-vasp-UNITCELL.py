#!/usr/bin/env runaiida
"""Launch script for the MagneticExchangeWorkChain."""
import numpy as np

from aiida import orm
from aiida.engine import submit, calcfunction
from aiida_common_workflows.common import ElectronicType
import ase.io
from ase.geometry import find_mic

from aiida_fourstate import MagneticExchangeWorkChain


### EIGER.ALPS
code_label = 'vasp-6.5.0-std@eiger.alps'
num_machines = 1
num_mpiprocs_per_machine = 128


protocol = 'custom'
#kpoints = [4, 4, 1]
print("ADAPT K-POINT GRID WHEN GOING TO 3x3x1 SUPERCELL!")
kpoints = [12, 12, 1] # For a single unit cell I use a denser grid


# LDA four-state v0 protocol for VASP
custom_protocol = {'name': 'fourstate-LDA-v1',
                    'description': 'Protocol for the LDA Four-state verification',
                    'potential_family': 'LDA.64',
                    # Here we still use the choice of which pseudos to use
                    # (how many semicore etc) as in the PBE. It's not so
                    # crucial, in practice it's Cr_pv and I.
                    'potential_mapping': 'RECOMMENDED_PBE',
                    'kpoints': kpoints,
                    'relax': {'algo': 'rd', 'threshold_forces': 0.001, 'steps': 200},
                    'parameters': {'prec': 'Accurate',
                                   'encut': 1000,
                                   'ediff': 1e-07,
                                   'ismear': -1,
                                   'sigma': 0.001, # 1 meV
                                   'algo': 'Normal',
                                   'nelmin': 6,
                                   'nelm': 300,
                                   'lmaxmix': 6,
                                   'lasph': True,
                                   'gga_compat': False,
                                   'ncore': 2,
                                   'kpar': 4}}

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


def launch_magnetic_exchange_calculation(neigh_idx = 1):
    """Launch a magnetic exchange coupling calculation.
    
    :param neigh_idx: Index of the neighbor to consider (1 means first neighbor)
    """
    global code_label, num_mpiprocs_per_machine

    supercell_matrix = [1, 1, 1] ## Setting this since it's already a supercell in input
    site1 = 0 # Cr

    print("FOR NOW, UNITCELL! REQUIRES SETTING CORRECTLY PARALLELIZATION PARAMS!")
    #ase_atoms = ase.io.read('cri3_3x3.cif')
    ase_atoms = ase.io.read('cri3_primitive.cif')
    structure_unitcell = orm.StructureData(ase=ase_atoms)

    if supercell_matrix == [1, 1, 1]:
        structure_supercell = structure_unitcell
    else:
        # Create the supercell structure
        structure_supercell = create_supercell(structure_unitcell, supercell_matrix)
    
    print(f"Unit cell has {len(structure_unitcell.sites)} atoms")
    print(f"Supercell has {len(structure_supercell.sites)} atoms")
    print(structure_supercell)

    site2, dist = find_neighbor(site1 = site1, neigh_idx=neigh_idx, filter_element="Cr", atoms=structure_supercell.get_ase())

    magnetization_per_site = [3. if structure_supercell.get_kind(kind_name).symbol == "Cr" else 0. for kind_name in structure_supercell.get_site_kindnames()]
    magnetization_magnitude = 3.0 # For the two sites to flip

    print(f"{magnetization_per_site=}")
    
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
        'engine_name': orm.Str('vasp'),
    }

    print(f"Submitting J({neigh_idx}) for CrI3 to {code_label} on {num_machines} node(s) with {num_mpiprocs_per_machine=}.")
    print("Submit? [Ctrl+C to stop]")
    input()
    
    print("Submitting MagneticExchangeWorkChain...")
    node = submit(MagneticExchangeWorkChain, **inputs)
    print(f"\nSubmitted: J({neigh_idx}), PK = {node.pk}")
    node.label = f"CrI3 J({neigh_idx}) calculation"
    node.base.extras.set('magnetic_exchange_neigh_idx', neigh_idx)


if __name__ == '__main__':
    for neigh_idx in [1]:
        print(f"\n\nLaunching VASP calculation of J({neigh_idx})*S^2\n")
        launch_magnetic_exchange_calculation(neigh_idx=neigh_idx)
