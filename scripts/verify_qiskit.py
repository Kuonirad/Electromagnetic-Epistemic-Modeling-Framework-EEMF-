from qiskit_aer import Aer


def main():
    backends = Aer.backends(name="statevector", method="matrix_product_state")
    print("MPI Backends:", backends)


if __name__ == "__main__":
    main()
