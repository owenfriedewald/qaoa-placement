import unittest
from qiskit import QuantumCircuit
from acm_submission_memo import MemoizedPhaseCompiler
from invariant_placement import circuit_hash


class MemoTests(unittest.TestCase):
    def test_exact_key_copy_isolation_and_bounded_retention(self):
        calls=[]
        def compiler(c):calls.append(circuit_hash(c));return c.copy()
        memo=MemoizedPhaseCompiler(compiler,max_entries=2)
        c=QuantumCircuit(2);c.cx(0,1);c.rz(.371,1);c.cx(0,1)
        expected=circuit_hash(c)
        first=memo(c);first.x(0)
        self.assertEqual(circuit_hash(memo(c.copy())),expected)
        self.assertEqual((memo.hits,memo.misses),(1,1))
        angle=QuantumCircuit(2);angle.rz(.372,1);memo(angle)
        phase=c.copy();phase.global_phase=.5;memo(phase)
        self.assertEqual(len(memo.cache),2)
        self.assertEqual(circuit_hash(memo(c)),expected)
        self.assertEqual(len(calls),4)  # Original entry was evicted, then rebuilt.
        wider=QuantumCircuit(3);wider.cx(0,1);wider.rz(.371,1);wider.cx(0,1)
        self.assertEqual(memo(wider).num_qubits,3)
        wires=QuantumCircuit(2);wires.cx(1,0);wires.rz(.371,0);wires.cx(1,0)
        self.assertEqual(circuit_hash(memo(wires)),circuit_hash(wires))
        self.assertEqual(len(calls),6)
        invalid=QuantumCircuit(2);invalid.x(0)
        with self.assertRaises(AssertionError):memo(invalid)


if __name__=='__main__':unittest.main()
