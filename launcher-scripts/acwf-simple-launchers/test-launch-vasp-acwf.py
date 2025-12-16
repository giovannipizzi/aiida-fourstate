#!/usr/bin/env runaiida
from ase.build import bulk

from aiida.orm import Code, StructureData
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

RelaxWorkChain = WorkflowFactory('common_workflows.relax.vasp')

ase_structure = bulk("Si", "diamond", a=5.45) # Slightly off lattice param
structure = StructureData(ase=ase_structure)

code_label = 'vasp-6.5.0-std@eiger.alps'


custom_fast = {'name': 'custom-fast',
 'description': 'Protocol to relax a structure with low precision at minimal computational cost for testing purposes.',
 'potential_family': 'PBE.64',
 'potential_mapping': 'RECOMMENDED_PBE',
 'kpoint_distance': 0.25,
 'relax': {'algo': 'cg', 'threshold_forces': 0.1, 'steps': 200},
 'parameters': {'prec': 'Single',
  'ediff': 0.0001,
  'ismear': 0,
  'sigma': 0.2,
  'algo': 'Veryfast',
  'nelm': 2000}}

input_generator = RelaxWorkChain.get_input_generator()
builder = input_generator.get_builder(engines={
        'relax': {
            'code': code_label,
            'options': {
                'resources': {
                    'num_machines': 1,
                },
                'max_wallclock_seconds': 900,
            }
        }
    },
    structure=structure,
    protocol="custom",
    custom_protocol=custom_fast,
    relax_type="positions_cell",  # full relaxation
    spin_type="none",     # Si is non-magnetic
)

node = submit(builder)
print(f"Submitted CommonRelaxWorkChain<{node.pk}>")

