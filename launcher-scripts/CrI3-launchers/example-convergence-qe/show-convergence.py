#!/usr/bin/env runaiida
from aiida import orm

SHOW_PK = True

GROUP_NAME = 'FourStateProtocol-convergence/CrI3/LDA/quantum_espresso/1x1x1cell'
group = orm.Group.collection.get(label=GROUP_NAME)

J_values = {}

for node in group.nodes:
    cutoff_wfc = node.base.extras.get('cutoff_wfc')

    if not node.is_finished_ok:
        J_values[cutoff_wfc] = (None, node.pk)
        continue

    J_values[cutoff_wfc] = (node.outputs.exchange_coupling.value, node.pk)

print("# WFC cutoff convergence for CrI3 unit cell, J(1), 12x12x1 k-point grid (ecutrho = 4x ecutwfc)")
    
print("# Cutoff[Ry]  J[meV]")
for cutoff in sorted(J_values):
    J, PK = J_values[cutoff]
    PK_string = f" # {PK}" if SHOW_PK else ''
    if J is None:
        print(f'#{cutoff}, [RUNNING OR ERROR]{PK_string}')
    else:
        print(f'{cutoff}, {J*1000.:.2f}{PK_string}')
