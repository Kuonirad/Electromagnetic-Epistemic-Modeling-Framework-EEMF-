from mpi4py import MPI

version, subversion = MPI.Get_version()
print(f"MPI {version}.{subversion} detected")
