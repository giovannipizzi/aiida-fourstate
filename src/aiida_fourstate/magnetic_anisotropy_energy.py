"""Workchain to compute magnetic anisotropy energy between
two different magnetization directions."""
from aiida import orm
from aiida.engine import WorkChain, calcfunction
from aiida_common_workflows.common.types import RelaxType, SpinType, ElectronicType

from aiida_common_workflows.plugins import load_workflow_entry_point


def validate_inputs(inputs, _):
    from numbers import Integral

    for site_idx in inputs['magnetic_sites']:
        if not isinstance(site_idx, Integral):
            return f'Site indices must be integer, instead found index of type {type(site_idx)}.'

    for site_idx in inputs['magnetic_sites']:
        if site_idx < 0:
            return f'Site indices must be >= 0, found site with index {site_idx}.'

    num_sites = len(inputs['structure'].sites)
    for site_idx in inputs['magnetic_sites']:
        if site_idx >= num_sites:
            return f'Site indices must be < num_sites = {num_sites}, found site with index {site_idx}.'


def validate_cartesian_3d_direction(vector, _):
    from numbers import Real
    import numpy as np

    if not len(vector) == 3:
        return 'The input vector should have exactly 3 components.'
    for x in vector:
        if not isinstance(x, Real):
            return 'Each component of the vector should be a Real number.'
    if not np.isclose(np.linalg.norm(vector), 1.):
        return 'The input vector should have unit length.'


@calcfunction
def compute_magnetic_anisotropy_energy(
    energy1, energy2, magnetic_sites,
):
    """Compute the magnetic anisotropy energy (mae) as the difference
    of the two supplied energies, per magnetic site.

    Uses the formula: mae = (e1 - e2) / n
    where:
        e1: Energy along first direction
        e2: Energy along second direction
        n: number of magnetic sites

    :return: Float node with MAE value in eV
    """
    e1 = energy1.value
    e2 = energy2.value
    n = len(magnetic_sites)

    mae = (e1 - e2) / n

    return orm.Float(mae)


class MagneticAnisotropyEnergyWorkChain(WorkChain):
    """Workchain to compute magnetic anisotropy energy between two different magnetization
    directions.

    This workchain computes the magnetic anistropy energy (mae) using two non-collinear
    spin configurations supplied by the user.

    Uses the formula: mae = (e1 - e2) / n
    where:
        e1: Energy along first direction
        e2: Energy along second direction
        n: number of magnetic sites
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData,
                   help='Input structure')
        spec.input('dir1', valid_type=orm.List,
                   validator=validate_cartesian_3d_direction,
                   required=False, default=lambda: orm.List([0., 0., 1.]),
                   help='First direction of the magnetization (Cartesian vector of unit length).')
        spec.input('dir2', valid_type=orm.List,
                   validator=validate_cartesian_3d_direction,
                   required=False, default=lambda: orm.List([1., 0., 0.]),
                   help='Second direction of the magnetization (Cartesian vector of unit length).')
        spec.input('magnetization_magnitude', valid_type=orm.Float,
                   required=False, default=lambda: orm.Float(1.0),
                   help='Magnitude of magnetization to set on magnetic sites (in µB)')
        spec.input('magnetic_sites', valid_type=orm.List,
                   help='List of magnetic sites in the structure.')
        spec.input('engine_name', valid_type=orm.Str)
        spec.inputs.validator = validate_inputs

        spec.input_namespace('generator_inputs',
            help='The inputs that will be passed to the input generator of the specified `sub_process`.')
        spec.input('generator_inputs.engines', valid_type=dict, non_db=True)
        spec.input('generator_inputs.custom_protocol', valid_type=dict, non_db=True, required=False, 
            help= 'Custom protocol dictionary to override default protocol settings, used if the protocol is set to "custom".', default=None)
        spec.input('generator_inputs.protocol', valid_type=str, non_db=True,
            help='The protocol to use when determining the workchain inputs.')
        spec.input('generator_inputs.electronic_type', valid_type=(ElectronicType, str), required=False, non_db=True,
            help='The type of electronics (insulator/metal) for the calculation.')
        spec.input('generator_inputs.threshold_forces', valid_type=float, required=False, non_db=True,
            help='Target threshold for the forces in eV/Å.')
        spec.input('generator_inputs.threshold_stress', valid_type=float, required=False, non_db=True,
            help='Target threshold for the stress in eV/Å^3.')
        # Code-dependent overrides for the sub_process
        spec.input_namespace('sub_process', dynamic=True, populate_defaults=False)

        spec.outline(
            cls.setup,
            cls.run_calculations,
            cls.inspect_calculations,
            cls.compute_mae,
            cls.results,
        )

        spec.output('magnetic_anisotropy_energy', valid_type=orm.Float,
                   help='Magnetic anisotropy energy in eV')
        spec.output('energy_dir1', valid_type=orm.Float,
                   help='Energy with spins along first direction.')
        spec.output('energy_dir2', valid_type=orm.Float,
                   help='Energy with spins along second direction.')

        spec.exit_code(401, 'ERROR_INVALID_SITE_INDICES',
                      message='Invalid site indices provided')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED',
                      message='One or more relaxation calculations failed')

    def setup(self):
        """Initialize context variables."""
        self.ctx.dir1 = self.inputs.dir1
        self.ctx.dir2 = self.inputs.dir2
        self.ctx.mag_magnitude = self.inputs.magnetization_magnitude.value
        self.ctx.sites = self.inputs.magnetic_sites

        # Do some reporting
        num_sites = len(self.inputs.structure.sites)
        num_magnetic = len(self.ctx.sites)
        self.report(f'Structure has {num_sites} atoms of which {num_magnetic} magnetic sites.')

    def run_calculations(self):
        """Submit two calculations with different magnetic configurations."""
        RelaxWorkChain = load_workflow_entry_point('relax', self.inputs.engine_name.value)

        for i, mag in enumerate([self.ctx.dir1, self.ctx.dir2]):
            inputs = self._build_relax_inputs(mag)
            generator = RelaxWorkChain.get_input_generator()
            builder = generator.get_builder(**inputs)
            # Apply any code-dependent overrides
            builder._merge(**self.inputs.get('sub_process', {}))

            # Assign a label for easier querability
            config_name = f'dir{i+1}'
            builder.metadata.call_link_label = config_name
            future = self.submit(builder)
            self.to_context(**{f'relax_{config_name}': future})
            self.report(f'Submitted {config_name} calculation (mag={str(mag)}).')

    def _build_relax_inputs(self, mag):
        """Build inputs for a RelaxWorkChain with specified magnetizations.

        :param mag: Magnetization (Cartesian vector in bohr magnetons)
        """
        generator_inputs = dict(self.inputs.generator_inputs)

        # Force spin_type to spin-orbit
        generator_inputs['spin_type'] = SpinType.SPIN_ORBIT.value
        # Remove any existing magnetization settings that we want to control
        generator_inputs.pop('fixed_total_cell_magnetization', None)

        # Set magnetization_per_site
        num_sites = len(self.inputs.structure.sites)

        # We start from a zero magnetization per site and only change the supplied magnetic sites
        magnetization_per_site = [0.0] * num_sites
        assert len(magnetization_per_site) == num_sites
        for site_idx in self.ctx.sites:
            magnetization_per_site[site_idx] = [x * self.ctx.mag_magnitude for x in mag]
        generator_inputs['magnetization_per_site'] = magnetization_per_site

        # This ensures both calculations use the same cell AND atom positions (no relaxation)
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
        for config_name in [f'dir{i+1}' for i in range(2)]:
            calc = self.ctx[f'relax_{config_name}']
            if not calc.is_finished_ok:
                failed.append(config_name)
                self.report(f'Calculation {config_name} failed with status: {calc.exit_status}')

        if failed:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED

    def compute_mae(self):
        """Extract energies and compute the exchange coupling constant."""

        # Extract total energies from each calculation
        try:
            self.ctx.energy1 = self._extract_energy(self.ctx.relax_dir1)
            self.ctx.energy2 = self._extract_energy(self.ctx.relax_dir2)
        except ValueError as e:
            self.report(f'Error extracting energy: {e}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED

        self.report(f'Energy first direction: {self.ctx.energy1.value:.6f} eV')
        self.report(f'Energy second direction: {self.ctx.energy2.value:.6f} eV')

        # Compute mae
        self.ctx.mae = compute_magnetic_anisotropy_energy(
            self.ctx.energy1,
            self.ctx.energy2,
            self.ctx.sites,
        )

        self.report(f'Magnetic anisotropy energy = {self.ctx.mae.value:.6f} eV')

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
        self.out('magnetic_anisotropy_energy', self.ctx.mae)
        self.out('energy_dir1', self.ctx.energy1)
        self.out('energy_dir2', self.ctx.energy2)

        self.report('MagneticAnisotropyEnergyWorkChain completed successfully')
