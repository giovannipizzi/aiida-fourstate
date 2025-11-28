#!/usr/bin/env runaiida
from pathlib import Path
from aiida import orm
from aiida.orm import load_node
from io import StringIO
import numpy as np
import tempfile
from collections import defaultdict

def get_output_lines(calc, transport):
    if transport is None:
        # Go via StringIO in order to have the same treatment of final \n
        # Using string.splitlines(), the \n are instead removed
        lines = StringIO(calc.outputs.retrieved.get_object_content('aiida.out')).readlines()
        return lines
    else:
        remote = calc.outputs.remote_folder
        remote_path = Path(remote.get_remote_path()) / 'aiida.out'
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                local_path = Path(tmpdirname) / 'temp.txt'
                transport.getfile(remote_path, local_path)
                with open(local_path) as fh:
                    lines = fh.readlines()
            return lines
        except FileNotFoundError:
            return None

def get_info(pk):
    wc = load_node(pk)
    site1 = wc.inputs.site1.value
    site2 = wc.inputs.site2.value

    is_abinit = wc.inputs.engine_name.value == 'abinit'

    NAMES = ['uu', 'ud', 'du', 'dd']
    if is_abinit:
        qb = orm.QueryBuilder()
        qb.append(orm.Node, filters={'id': wc.pk}, tag='commonwf4state')
        qb.append(orm.WorkChainNode, with_incoming='commonwf4state', tag='commonrelax')
        qb.append(orm.WorkChainNode, with_incoming='commonrelax', tag='abinitbase', project='*')
        qb.append(orm.CalcJobNode, with_incoming='abinitbase', tag='abinitcalc', project='*')
        calcjobs_raw = list(qb.all())
        calcjobs_unsorted = defaultdict(list)
        for w, c in calcjobs_raw:
            calcjobs_unsorted[w.pk].append(c)

        #print(calcjobs_unsorted)

        # convert just in a list of lists, but sort correctly by type of calculation
        # TODO: in the future, adapt the submission script to use the CALL labels
        # so we can remove this logic!
        wf_to_state_mapping = {}
        for wpk, cjs in calcjobs_unsorted.items():
            cj = cjs[0]
            starting_mag_at_sites = np.array(cj.inputs.parameters.get_dict()['spinat'])[:,2]
            positive_count = len(starting_mag_at_sites[starting_mag_at_sites>1.e-6])
            negative_count = len(starting_mag_at_sites[starting_mag_at_sites<-1.e-6])

            if negative_count == 0:
                assert 'uu' not in wf_to_state_mapping
                wf_to_state_mapping['uu'] = wpk
            elif negative_count == 1:
                if 'ud' in wf_to_state_mapping:
                    assert 'du' not in wf_to_state_mapping
                    wf_to_state_mapping['du'] = wpk
                wf_to_state_mapping['ud'] = wpk
            elif negative_count == 2:
                assert 'dd' not in wf_to_state_mapping
                wf_to_state_mapping['dd'] = wpk
            else:
                raise AssertionError(f"Unexpected state! {positive_count=}, {negative_count=}")

        assert set(wf_to_state_mapping) == set(NAMES)
        assert len(wf_to_state_mapping) == len(NAMES)

        calcjobs = [calcjobs_unsorted[wf_to_state_mapping[label]] for label in NAMES]
    else: # QE
        qb = orm.QueryBuilder()
        qb.append(orm.Node, filters={'id': wc.pk}, tag='commonwf4state')
        qb.append(orm.WorkChainNode, with_incoming='commonwf4state', tag='commonrelax')
        qb.append(orm.WorkChainNode, with_incoming='commonrelax', tag='qerelax')
        qb.append(orm.WorkChainNode, with_incoming='qerelax', tag='pwbase', project='*')
        qb.append(orm.CalcJobNode, with_incoming='pwbase', tag='pwcalc', project='*')
        calcjobs_raw = list(qb.all())
        calcjobs_unsorted = defaultdict(list)
        for w, c in calcjobs_raw:
            calcjobs_unsorted[w.pk].append(c)

        # TODO: in the future, adapt the submission script to use the CALL labels
        # so we can remove this logic! possibly unify with above
        wf_to_state_mapping = {}
        for wpk, cjs in calcjobs_unsorted.items():
            cj = cjs[0]
            starting_magnetizations = cj.inputs.parameters.get_dict()['SYSTEM']['starting_magnetization']
            kinds_at_sites = cj.inputs.structure.get_site_kindnames()
            starting_mag_at_sites = np.array([starting_magnetizations[k] for k in kinds_at_sites])
            positive_count = len(starting_mag_at_sites[starting_mag_at_sites>1.e-6])
            negative_count = len(starting_mag_at_sites[starting_mag_at_sites<-1.e-6])

            if negative_count == 0:
                assert 'uu' not in wf_to_state_mapping
                wf_to_state_mapping['uu'] = wpk
            elif negative_count == 1:
                if 'ud' in wf_to_state_mapping:
                    assert 'du' not in wf_to_state_mapping
                    wf_to_state_mapping['du'] = wpk
                wf_to_state_mapping['ud'] = wpk
            elif negative_count == 2:
                assert 'dd' not in wf_to_state_mapping
                wf_to_state_mapping['dd'] = wpk
            else:
                raise AssertionError(f"Unexpected state! {positive_count=}, {negative_count=}")

        assert set(wf_to_state_mapping) == set(NAMES)
        assert len(wf_to_state_mapping) == len(NAMES)

        calcjobs = [calcjobs_unsorted[wf_to_state_mapping[label]] for label in NAMES]
        

    upup = calcjobs[0][-1]
    print(upup)

    output_files = {}
    calcs = {}
    if not wc.sealed:
        with upup.computer.get_transport() as transport:

            for idx, name in enumerate(NAMES):
                calc = calcjobs[idx][-1]
                calcs[name] = calc
                assert calc.computer == upup.computer

                lines = get_output_lines(calc, transport)
                output_files[name] = lines
    else:
        for idx, name in enumerate(NAMES):
            calc = calcjobs[idx][-1]
            calcs[name] = calc
            lines = get_output_lines(calc, transport=None)
            output_files[name] = lines

    for name in NAMES:
        print(f"PK[{name}] ==> PK = {calcs[name].pk}; remote_folder ==> PK = {calcs[name].outputs.remote_folder.pk}")

        lines = output_files[name]
        if lines is None:
            print("  [no output generated yet!]")
            print()
        else:
            # Only show total energy if not yet finished or failed
            if not calcs[name].is_finished_ok:
                lines_idx = [idx for idx, l in enumerate(lines) if 'accuracy' in l][-2:]
                lines_idx2 = []
                for idx in lines_idx:
                    if idx > 0:
                        lines_idx2.append(idx-1)
                    lines_idx2.append(idx)
                lines_idx2 = set(lines_idx2)
                acc_lines = [l for idx, l in enumerate(lines) if  idx in lines_idx2]
                print("".join(acc_lines))
                print()

            print("  Magnetizations summary:")
            if is_abinit:
                print("".join([l for idx, l in enumerate(lines) if 'magnetiz' in l][-3:-1]))
            else:
                print("".join([l for idx, l in enumerate(lines) if 'magnetiz' in l][-2:]))
                
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

            try:
                # Trying to pick outputs first because in earlier versions of the aiida-abinit plugin,
                # the code is going back to primitive cell internally, so the input cell might be
                # a supercell but the code sees a unit cell
                kinds_at_site = np.array(calcs[name].outputs.output_structure.get_site_kindnames() )
            except AttributeError:
                # In case e.g. the code is still running
                kinds_at_site = np.array(calcs[name].inputs.structure.get_site_kindnames() )

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

