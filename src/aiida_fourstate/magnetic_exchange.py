"""Workchain to compute magnetic exchange coupling J using the 4-state method."""
from aiida import orm
from aiida.engine import WorkChain, ToContext, calcfunction
from aiida_common_workflows.common.types import RelaxType, SpinType, ElectronicType

from aiida_common_workflows.plugins import load_workflow_entry_point


@calcfunction
def compute_exchange_coupling(
    energy_upup, energy_updown, energy_downup, energy_downdown
):
    """Compute the exchange coupling constant J*S^2 from the four energies.
    
    Uses the formula: J*S^2 = (E_AFM_avg - E_FM_avg) / 2
    where:
        E_FM_avg = (E_upup + E_downdown) / 2
        E_AFM_avg = (E_updown + E_downup) / 2
    
    :return: Float node with J*S^2 value in eV
    """
    e_upup = energy_upup.value
    e_updown = energy_updown.value
    e_downup = energy_downup.value
    e_downdown = energy_downdown.value
    
    e_fm_avg = (e_upup + e_downdown) / 2.0
    e_afm_avg = (e_updown + e_downup) / 2.0
    
    j_times_s2 = (e_afm_avg - e_fm_avg) / 2.0
    
    return orm.Float(j_times_s2)


class MagneticExchangeWorkChain(WorkChain):
    """Workchain to compute magnetic exchange coupling using the 4-state method.
    
    This workchain computes the exchange coupling constant J between two magnetic sites
    using four collinear spin configurations: up-up, up-down, down-up, down-down.
    """
    
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        
        spec.input('structure', valid_type=orm.StructureData,
                   help='Input supercell structure')
        spec.input('site1', valid_type=orm.Int,
                   help='Index of the first magnetic site in the supercell (0-based)')
        spec.input('site2', valid_type=orm.Int,
                   help='Index of the second magnetic site in the supercell (0-based)')
        spec.input('magnetization_magnitude', valid_type=orm.Float,
                   required=False, default=lambda: orm.Float(1.0),
                   help='Magnitude of magnetization to set on magnetic sites (in µB)')
        spec.input('engine_name', valid_type=orm.Str)
        
        spec.input_namespace('generator_inputs',
            help='The inputs that will be passed to the input generator of the specified `sub_process`.')
        spec.input('generator_inputs.engines', valid_type=dict, non_db=True)
        spec.input('generator_inputs.custom_protocol', valid_type=dict, non_db=True, required=False, 
            help= 'Custom protocol dictionary to override default protocol settings, used if the protocol is set to "custom".', default=None)
        spec.input('generator_inputs.protocol', valid_type=str, non_db=True,
            help='The protocol to use when determining the workchain inputs.')
        spec.input('generator_inputs.electronic_type', valid_type=(ElectronicType, str), required=False, non_db=True,
            help='The type of electronics (insulator/metal) for the calculation.')
        spec.input('generator_inputs.magnetization_per_site', valid_type=(list, tuple), required=False, non_db=True,
            help='List containing the initial magnetization per atomic site.')        
        spec.input('generator_inputs.threshold_forces', valid_type=float, required=False, non_db=True,
            help='Target threshold for the forces in eV/Å.')
        spec.input('generator_inputs.threshold_stress', valid_type=float, required=False, non_db=True,
            help='Target threshold for the stress in eV/Å^3.')
        # Code-dependent overrides for the sub_process
        spec.input_namespace('sub_process', dynamic=True, populate_defaults=False)

        spec.outline(
            cls.setup,
            cls.run_four_states,
            cls.inspect_calculations,
            cls.compute_j_coupling,
            cls.results,
        )
        
        spec.output('exchange_coupling', valid_type=orm.Float,
                   help='Exchange coupling constant J*S^2 in eV')
        spec.output('energy_upup', valid_type=orm.Float,
                   help='Energy of up-up configuration')
        spec.output('energy_updown', valid_type=orm.Float,
                   help='Energy of up-down configuration')
        spec.output('energy_downup', valid_type=orm.Float,
                   help='Energy of down-up configuration')
        spec.output('energy_downdown', valid_type=orm.Float,
                   help='Energy of down-down configuration')
        
        spec.exit_code(401, 'ERROR_INVALID_SITE_INDICES',
                      message='Invalid site indices provided')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED',
                      message='One or more relaxation calculations failed')
    
    def setup(self):
        """Initialize context variables."""
        self.ctx.site1_idx = self.inputs.site1.value
        self.ctx.site2_idx = self.inputs.site2.value
        self.ctx.mag_magnitude = self.inputs.magnetization_magnitude.value
        
        # Validate inputs
        if self.ctx.site1_idx < 0 or self.ctx.site2_idx < 0:
            self.report(f'Site indices must be > 0, they are {self.ctx.site1_idx} and {self.ctx.site2_idx}')
            return self.exit_codes.ERROR_INVALID_SITE_INDICES
        
        num_sites = len(self.inputs.structure.sites)
        if self.ctx.site1_idx >= num_sites or self.ctx.site2_idx >= num_sites:
            self.report(f'Site indices must be < num_sites = {num_sites}, they are {self.ctx.site1_idx} and {self.ctx.site2_idx}')
            return self.exit_codes.ERROR_INVALID_SITE_INDICES
        
        self.report(f'(Supercell) structure has {num_sites} atoms')
        self.report(f'Site1 index: {self.ctx.site1_idx}')
        self.report(f'Site2 index: {self.ctx.site2_idx}')
    
    def run_four_states(self):
        """Submit four calculations with different magnetic configurations."""
        RelaxWorkChain = load_workflow_entry_point('relax', self.inputs.engine_name.value)

        # TODO: Move to class method, reuse later
        configurations = {
            'UU': (1.0, 1.0),
            'UD': (1.0, -1.0),
            'DU': (-1.0, 1.0),
            'DD': (-1.0, -1.0),
        }
        
        for config_name, (mag1, mag2) in configurations.items():
            inputs = self._build_relax_inputs(mag1, mag2)
            generator = RelaxWorkChain.get_input_generator()
            builder = generator.get_builder(**inputs)
            # Apply any code-dependent overrides
            builder._merge(**self.inputs.get('sub_process', {}))

            # Assign a label for easier querability
            builder.metadata.call_link_label = config_name
            future = self.submit(builder)
            self.to_context(**{f'relax_{config_name}': future})
            self.report(f'Submitted {config_name} calculation (up={mag1:+.1f}, down={mag2:+.1f})')
    
    def _build_relax_inputs(self, mag1, mag2):
        """Build inputs for a RelaxWorkChain with specified magnetizations.
        
        :param mag1: Magnetization for site1 (in bohr magnetons)
        :param mag2: Magnetization for site2 (in bohr magnetons)
        """
        generator_inputs = dict(self.inputs.generator_inputs)
        
        # Force spin_type to collinear
        generator_inputs['spin_type'] = SpinType.COLLINEAR.value
        # Remove any existing magnetization settings that we want to control
        generator_inputs.pop('fixed_total_cell_magnetization', None)

        # Set magnetization_per_site
        num_sites = len(self.inputs.structure.sites)

        # We start from a given magnetization per site (given in input, if given), and only change the two sites
        magnetization_per_site = generator_inputs.get('magnetization_per_site', [0.0] * num_sites)
        assert len(magnetization_per_site) == num_sites
        magnetization_per_site[self.ctx.site1_idx] = mag1 * self.ctx.mag_magnitude
        magnetization_per_site[self.ctx.site2_idx] = mag2 * self.ctx.mag_magnitude
        generator_inputs['magnetization_per_site'] = magnetization_per_site

        # This ensures all four calculations use the same cell AND atom positions (no relaxation)
        generator_inputs['relax_type'] = RelaxType.NONE.value

        inputs = {
            'structure': self.inputs.structure,
        }
        inputs.update(generator_inputs)
        return inputs
    
    def inspect_calculations(self):
        """Check that all calculations finished successfully."""
        failed = []
        
        # TODO: reuse list from class method refactored above
        for config_name in ['upup', 'updown', 'downup', 'downdown']:
            calc = self.ctx[f'relax_{config_name}']
            if not calc.is_finished_ok:
                failed.append(config_name)
                self.report(f'Calculation {config_name} failed with status: {calc.exit_status}')
        
        if failed:
            self.report(f'The following calculations failed: {", ".join(failed)}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
    
    def compute_j_coupling(self):
        """Extract energies and compute the exchange coupling constant."""
        
        # Extract total energies from each calculation
        try:
            self.ctx.energy_upup = self._extract_energy(self.ctx.relax_upup)
            self.ctx.energy_updown = self._extract_energy(self.ctx.relax_updown)
            self.ctx.energy_downup = self._extract_energy(self.ctx.relax_downup)
            self.ctx.energy_downdown = self._extract_energy(self.ctx.relax_downdown)
        except ValueError as e:
            self.report(f'Error extracting energy: {e}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED

        self.report(f'Energy up-up: {self.ctx.energy_upup.value:.6f} eV')
        self.report(f'Energy up-down: {self.ctx.energy_updown.value:.6f} eV')
        self.report(f'Energy down-up: {self.ctx.energy_downup.value:.6f} eV')
        self.report(f'Energy down-down: {self.ctx.energy_downdown.value:.6f} eV')
        
        # Compute J*S^2
        self.ctx.j_coupling = compute_exchange_coupling(
            self.ctx.energy_upup,
            self.ctx.energy_updown,
            self.ctx.energy_downup,
            self.ctx.energy_downdown
        )
        
        self.report(f'Exchange coupling J*S^2 = {self.ctx.j_coupling.value:.6f} eV')
    
    def _extract_energy(self, workchain):
        """Extract the total energy from a completed RelaxWorkChain.
        
        :param workchain: Completed RelaxWorkChain
        :return: Float node with energy in eV
        """
        # The energy should be in the output namespace
        # Common workflows typically output 'total_energy'
        if 'total_energy' in workchain.outputs:
            return workchain.outputs.total_energy
        
        raise ValueError('Could not extract energy from RelaxWorkChain outputs')
    
    def results(self):
        """Store results in outputs."""
        self.out('exchange_coupling', self.ctx.j_coupling)
        self.out('energy_upup', self.ctx.energy_upup)
        self.out('energy_updown', self.ctx.energy_updown)
        self.out('energy_downup', self.ctx.energy_downup)
        self.out('energy_downdown', self.ctx.energy_downdown)
        
        self.report('MagneticExchangeWorkChain completed successfully')
