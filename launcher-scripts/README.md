# Launcher scripts for the four-state workflow

This folder includes:
- simple examples to run a Si relaxation via the common workflows (folder `acwf-simple-launchers`), to easily test the functionality and the code on the remote (both for Quantum ESPRESSO and ABINIT)
- some example computer and code configuration files (YAML for AiiDA), that I used myself. They require of course adaptation, but could turn out useful for others as a template.
- example launcher scripts (adapted for Quantum ESPRESSO and ABINIT) for CrI3 (folder `CrI3-launchers`)
- A script `./inspect-results.py` that, both for QE and ABINIT, tried to fetch information of a given submitted `MagneticExchangeWorkChain` (this tries also to provide results while the simulation is still running). Could (and should) be simplified in the future.