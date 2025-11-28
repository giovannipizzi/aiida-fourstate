#!/usr/bin/env runaiida
from ase.build import bulk

from aiida.orm import Code, StructureData
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

RelaxWorkChain = WorkflowFactory('common_workflows.relax.quantum_espresso')

ase_structure = bulk("Si", "diamond", a=5.45) # Slightly off lattice param
structure = StructureData(ase=ase_structure)



#code_label = 'abinit@xxx'
code_label = 'pw-7.4@thor'

input_generator = RelaxWorkChain.get_input_generator()
builder = input_generator.get_builder(engines={
        'relax': {
            'code': code_label,
            'options': {
                'resources': {
                    'num_machines': 1,
                },
                'max_wallclock_seconds': 3600,
            }
        }
    },
    structure=structure,
    protocol="fast",      # or "moderate", "precise"
    relax_type="positions_cell",  # full relaxation
    spin_type="none",     # Si is non-magnetic
)

node = submit(builder)
print(f"Submitted CommonRelaxWorkChain<{node.pk}>")

