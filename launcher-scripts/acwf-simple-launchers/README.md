# How to launch a simple calculation

## ABINIT
- Install the code on a supercomputer (I could install it with micromamba)
- Install this package and its dependencies (including aiida-common-workflows)
- Install the pseudos, e.g. `aiida-pseudo install pseudo-dojo -f jthxml -v 1.0`
- Setup the computer. E.g. in my case I'm using the configuration files in `../aiida-config-examples/abinit/` and setting up with
  - `verdi computer setup --config thor-micromamba-setup.yaml`
  - `verdi -p presto computer configure core.ssh thor-micromamba --config thor-micromamba-config.yaml`
  - `verdi computer test thor-micromamba`
  - `verdi code create core.code.installed --config abinit-10.0.3-thor-micromamba.yaml`

