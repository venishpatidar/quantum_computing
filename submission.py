import pennylane as qml

DEVICE = "default.qubit"

class PennylaneIntro:
    def __init__(self):
        """Intro to Pennylane Questions Class"""
        
    def get_device(self,wires:int):
        """ Create a pennylane default.qubit device with multiple qubits (equal to wires)
        Args:
            wires (int): Number of Qubits in the circuit

        Returns:
            qml.device: Pennylane Device
        """
        dev = qml.device(DEVICE, wires=wires)  # Define Device Here
        return dev

    def simple_circuit(self) -> qml.numpy.tensor:
        """ Simple circuit applying a Hadamard Gate on a single qubit syste. (The device is already created, only code the circuit)

        Returns:
            qml.numpy.tensor: Output State of the functions
        """
        qml.Hadamard(wires=0)
        return qml.state()

    def probability_circuit(self) -> qml.numpy.tensor:
        """ Circuit returning the probabilities of measuring the state in the |0> and |1> states after applying a Hadamard Gate.
        Returns:
            qml.numpy.tensor: Tensor containing the probabilities of measuring the state in the |0> and |1> states
        """
        qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    def create_qnode(self) -> qml.QNode:
        """ Use the device from the get_device function and create a QNode for the probability_circuit function

        Returns:
            qml.QNode: return the QNode object
        """
        dev = self.get_device(1)
        qnode = qml.QNode(self.probability_circuit, dev)
        return qnode

    def create_qnode_decorator(self) -> qml.QNode:
        """ Use the device from the get_device function with a single qubit and create a QNode for the probability_circuit function

        Returns:
            qml.QNode: return the QNode object
        """
        dev = self.get_device(1)
        @qml.qnode(dev)
        def qnode():
            qml.Hadamard(wires=0)
            return qml.probs(wires=0)
        return qnode


class IntermediateQubitSystems:
    def __init__(self):
        """Intermediate qubit Questions Class"""

    def create_classically_controlled_circuit(self) -> qml.QNode:
        """ Implements a circuit which returns ket 0 if the number of 1's is even and ket 1 if the number of 1's is odd

        Returns:
            qml.QNode: return the QNode object
        """
        dev = qml.device(DEVICE, wires=0)  # Define a single qubit device here
        @qml.qnode(dev)
        def qnode(bitstring:int):
            """ Implements a circuit which returns ket 0 if the number of 1's is even and ket 1 if the number of 1's is odd

            Args:
                bitstring (str): Binary string of 0's and 1's
            Returns:
                state: qubit state
            """
            for bit in str(bitstring):
                if bit=="1":
                    qml.PauliX(wires=0)
            return qml.state()
        return qnode

    def create_compliled_circuit(self,gates) -> qml.QNode:
        """ Given a set of unparameterized gates in the order they are applied, create a QNode that applies the gates to a single qubit system

        Args:
            gates (list): List of pennylane gates to be applied to the qubit
        Returns:
            qml.QNode: return the QNode object
        """
        dev = qml.device(DEVICE, wires=0)  # Define a single qubit device here
        U = qml.math.eye(2)  # Replace with corect Unitary Matrix
        for gate in gates:
            U = gate(wires=0).matrix() @ U  # Apply matrix multiplication

        @qml.qnode(dev)
        def qnode():
            """ Classically Controlled QNode that appplies a Hadamard Gate on a single qubit system based on the initial state
            Returns:
                state: qubit state
            """
            qml.QubitUnitary(U, wires=0)
            return qml.state()
        return qnode
