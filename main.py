import unittest
from submission import PennylaneIntro, IntermediateQubitSystems 
import pennylane as qml
import numpy as np

class TestPennylaneIntro(unittest.TestCase):
    """Unit tests for the PennylaneIntro class."""

    def setUp(self):
        """Initialize PennylaneIntro instance before each test."""
        self.pl = PennylaneIntro()
        self.iqs = IntermediateQubitSystems()
                
    def test_create_qnode_decorated(self):
        """Test if create_qnode_decorator returns a valid QNode"""
        qnode = self.pl.create_qnode_decorated()
        self.assertIsInstance(qnode, qml.QNode)
        # Execute the QNode and check output
        result = qnode()
        self.assertIsInstance(qnode(), np.ndarray)
        self.assertEqual(result.shape, (2,))  # Probabilities of a single qubit system
    
    def test_create_qnode(self):
        """Test if create_qnode returns a valid QNode"""
        qnode = self.pl.create_qnode()
        self.assertIsInstance(qnode, qml.QNode)
        # Execute the QNode and check output
        result = qnode()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))  # Since we expect probabilities for 1 qubit


    def test_create_classically_controlled_circuit(self):
        """Test if the function returns a valid QNode and behaves correctly"""
        qnode = self.iqs.create_classically_controlled_circuit()
        self.assertIsInstance(qnode, qml.QNode)
        self.assertEqual(len(qnode.device._wires), 1, "Circuit should use exactly 1 qubit")

        # Test with bitstrings
        even_parity_input = 1010  # Two 1's (even parity), should return [1, 0]
        odd_parity_input = 1101  # Three 1's (odd parity), should return [0, 1]

        result_even = qnode(even_parity_input)
        result_odd = qnode(odd_parity_input)
        # Expected output: |0⟩ corresponds to [1, 0], |1⟩ corresponds to [0, 1]
        np.testing.assert_array_almost_equal(result_even, [1, 0], decimal=5)
        np.testing.assert_array_almost_equal(result_odd, [0, 1], decimal=5)

    def test_create_compliled_circuit(self):
        """Test if the function returns a valid QNode and behaves correctly"""
        qnode = self.iqs.create_classically_controlled_circuit()
        self.assertIsInstance(qnode, qml.QNode)
        self.assertEqual(len(qnode.device._wires), 1, "Circuit should use exactly 1 qubit")

        # List of test gates
        gates_1 = [qml.Hadamard]
        gates_2 = [qml.PauliX, qml.PauliZ]

        # Get QNodes
        qnode_1 = self.iqs.create_compliled_circuit(gates_1)
        qnode_2 = self.iqs.create_compliled_circuit(gates_2)

        # Ensure QNode is returned
        self.assertIsInstance(qnode_1, qml.QNode)
        self.assertIsInstance(qnode_2, qml.QNode)

        # Test output state for Hadamard gate
        expected_state_hadamard = np.array([1, 1]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(qnode_1(), expected_state_hadamard, decimal=5)

        # Test output state for PauliX followed by PauliZ (Equivalent to -PauliX)
        expected_state_pauli_xz = np.array([0, -1])
        np.testing.assert_array_almost_equal(qnode_2(), expected_state_pauli_xz, decimal=5)

if __name__ == '__main__':
    unittest.main()
