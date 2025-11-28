#!/usr/bin/env runaiida
from pathlib import Path
from aiida import orm
from aiida.orm import load_node
from io import StringIO
import numpy as np
import tempfile
from collections import defaultdict

def get_output_lines(calc, transport):
    output_filename = calc.base.attributes.get('output_filename')
    if transport is None: # Get from AiiDA retrieved files
        # Go via StringIO in order to have the same treatment of final \n
        # Using string.splitlines(), the \n are instead removed
        lines = StringIO(calc.outputs.retrieved.get_object_content(
            output_filename)).readlines()
        return lines
    else: # Still running
        remote = calc.outputs.remote_folder
        remote_path = Path(remote.get_remote_path()) / output_filename
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                local_path = Path(tmpdirname) / 'temp.txt'
                transport.getfile(remote_path, local_path)
                with open(local_path) as fh:
                    lines = fh.readlines()
            return lines
        except FileNotFoundError:
            return None

def grep(lines, pattern, num_last_matches=None, lines_before=0, skip_last_matches=0):
    # Filter by pattern
    lines_idx = [idx for idx, l in enumerate(lines) if pattern in l]
    
    # Get only the last `num_last_matches` matches
    if num_last_matches is not None:
        lines_idx = lines_idx[-num_last_matches:]
    if skip_last_matches is not None:
        lines_idx = lines_idx[:-skip_last_matches]

    # Include `lines_before` lines before matches
    lines_idx2 = []
    for idx in lines_idx:
        if idx > 0:
            for i in range(1, lines_before+1):
                lines_idx2.append(idx - i)
        lines_idx2.append(idx)
    lines_idx2 = set(lines_idx2)
    acc_lines = [l for idx, l in enumerate(lines) if  idx in lines_idx2]
    return acc_lines

def get_info(pk):
    wc = load_node(pk)

    is_abinit = wc.inputs.engine_name.value == 'abinit'

    NAMES = ['upup', 'updown', 'downup', 'downdown']
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={'id': wc.pk}, tag='commonwf4state')
    qb.append(orm.WorkChainNode, with_incoming='commonwf4state', tag='commonrelax', edge_project=['label'])
    if is_abinit:
        qb.append(orm.WorkChainNode, with_incoming='commonrelax', tag='abinitbase')
        qb.append(orm.CalcJobNode, with_incoming='abinitbase', tag='abinitcalc', project='*')
    else: # QE
        qb.append(orm.WorkChainNode, with_incoming='commonrelax', tag='qerelax')
        qb.append(orm.WorkChainNode, with_incoming='qerelax', tag='pwbase')
        qb.append(orm.CalcJobNode, with_incoming='pwbase', tag='pwcalc', project='*')

    # Group calcjobs by parent workflow label (upup, updown, ...)
    calcjobs = defaultdict(list)
    for calcjob, edge_label in qb.all():
        calcjobs[edge_label].append(calcjob)
    assert sorted(calcjobs.keys()) == sorted(NAMES), f"Unexpected workflow CALL link labels! {calcjobs.keys()=}"
    # Sort calcjobs chronologically within each group
    for edge_label in calcjobs:
        calcjobs[edge_label].sort(key=lambda calcjob: calcjob.ctime)
    print(calcjobs)        

    upup = calcjobs['upup'][-1] # The last upup calculation
    print(upup)

    def get_calcs_and_output_files(calcjobs, NAMES, transport=None):
        output_files = {}
        calcs = {}
        for name in NAMES:
            calc = calcjobs[name][-1]
            calcs[name] = calc
            assert calc.computer == upup.computer
            lines = get_output_lines(calc, transport)
            output_files[name] = lines

        return calcs, output_files

    if not wc.sealed:
        # If it's still running, use transport to get remote files directly from computer
        with upup.computer.get_transport() as transport:
            calcs, output_files = get_calcs_and_output_files(calcjobs, NAMES, transport)
    else:
        # If calculation is sealed (so, finished), do not open SSH connections but get from retrieved files in the AiiDA repo
        calcs, output_files = get_calcs_and_output_files(calcjobs, NAMES) # Do not pass transport here

    for name in NAMES:
        print(f"PK[{name}] ==> PK = {calcs[name].pk}; remote_folder ==> PK = {calcs[name].outputs.remote_folder.pk}")

        lines = output_files[name]
        if lines is None:
            print("  [no output generated yet!]")
            print()
        else:
            # Only show total energy if not yet finished or failed
            if not calcs[name].is_finished_ok:
                acc_lines = grep(lines, 'accuracy', num_last_matches=2, lines_before=1)
                print("".join(acc_lines))
                print()

            print("  Magnetizations summary:")
            if is_abinit:
                acc_lines = grep(lines, 'magnetiz', num_last_matches=3, skip_last_matches=1)                
            else:
                acc_lines = grep(lines, 'magnetiz', num_last_matches=2) 
            print("".join(acc_lines))

            if is_abinit:
                line_nums = [line_idx for line_idx, line in enumerate(lines) if 'ratsph' in line]

                if line_nums:
                    # Only one line should be found
                    assert len(line_nums) == 1
                    line_num = line_nums[0]
                    
                    # Check content of next line
                    assert 'Atom    Radius    up_density   dn_density  Total(up+dn)  Diff(up-dn)' == lines[line_num+1].strip(), f"Unexpected line content: {lines[line_num+1].strip()}"
                    magn_data = []
                    indices = []
                    num_atoms = len(calcs[name].outputs.output_structure.get_site_kindnames())
                    for idx in range(line_num+2, line_num+2+num_atoms):
                        pieces = lines[idx].split()
                        indices.append(int(pieces[0]))
                        magn_data.append(float(pieces[5])) # Diff(up-dn)
                    assert indices == list(range(1, num_atoms+1)), f"Wrong atom indices! {indices=}, expecting {num_atoms} atoms"
                    magn_data = np.array(magn_data)
                else:
                    magn_data = np.array([])
            else: #QE
                 magn_lines = [_ for _ in lines if '(R=' in _]
                 magn_data = [(int(l.split()[1]), float(l.split()[6])) for l in magn_lines]
                 assert [_[0] for _ in magn_data] == list(range(1, len(magn_data)+1))
                 # Only magnetizations
                 magn_data = np.array([_[1] for _ in magn_data])

            kinds_at_site = np.array(calcs[name].inputs.structure.get_site_kindnames())

            # Check relevant potentially magnetic kinds
            for kind_name in ['Ni0', 'Ni1', 'Ni', 'Cr0', 'Cr1', 'Cr']:
                if kind_name not in kinds_at_site:
                    continue
                if len(magn_data):
                    kind_magn = magn_data[kinds_at_site == kind_name]
                else:
                    kind_magn = []
                if is_abinit:
                    # Does not define 2 kinds, they are both in Cr
                    kind_magn_pos = np.array([k for k in kind_magn if k >= 0])
                    kind_magn_neg = np.array([k for k in kind_magn if k < 0])

                    for name, this_kind_magn in [
                            ('positive', kind_magn_pos),
                            ('negative', kind_magn_neg)]:
                        if len(this_kind_magn): # Not empty
                            min_kind = this_kind_magn.min()
                            max_kind = this_kind_magn.max()
                            num_sites = len(this_kind_magn)
                            site_s = '' if num_sites == 1 else 's'
                            print(f"     {kind_name} ({name}) magnetizations ({num_sites} site{site_s}): {(min_kind + max_kind)/2:.4f} ± {(max_kind - min_kind)/2:.4f}")
                else:
                    if len(kind_magn): # Not empty
                        min_kind = kind_magn.min()
                        max_kind = kind_magn.max()
                        num_sites = len(kind_magn)
                        site_s = '' if num_sites == 1 else 's'

                        print(f"     {kind_name} magnetizations ({num_sites} site{site_s}): {(min_kind + max_kind)/2:.4f} ± {(max_kind - min_kind)/2:.4f}")
            print()

    status = f" ({wc.process_status})" if wc.process_status else ""
    print(f"{wc.process_label} workflow (PK={wc.pk})")
    print(f"Current workflow state: {wc.process_state.value}{status}")

    try:
        print(f"Exchange coupling: {wc.outputs.exchange_coupling.value * 1000.:.2f} meV")
        print(f"  - up-up: {wc.outputs.energy_upup.value * 1000.:.2f} meV")
        print(f"  - up-dn: {wc.outputs.energy_updown.value * 1000.:.2f} meV")
        print(f"  - dn-up: {wc.outputs.energy_downup.value * 1000.:.2f} meV")
        print(f"  - dn-dn: {wc.outputs.energy_downdown.value * 1000.:.2f} meV")
    except AttributeError:
        print("[Final results not yet available]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Pass PK or UUID of MagneticExchangeWorkChain!")
        sys.exit(1)
    get_info(sys.argv[1])

