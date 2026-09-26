# Execution corrections (specification unchanged)

Preflight job 17007056, snapshot SHA-256
731530e4e0b0b45047916012e54d3c914d8873acbc58e830c759548a55383ab0,
passed three independent scientific checks and failed importing PyZX because
its new dependency lark was absent. No campaign cases ran. The isolated PyZX
0.10.3 package was installed with --no-deps to protect the established scientific
environment. The retry adds its required dependency to a separate target path;
it does not upgrade Qiskit, NumPy or SciPy. The failed run directory is retained.
The budget selection input path was also corrected before campaign execution
(from a child of the quality directory to its budget sibling); this changes no
selection rule or outcomes.

The environment-complete preflight, job 17007113, then exposed an OpenQASM
interchange issue: PyZX emits the standard swap name, which Qiskit's strict
parser does not register by default. The retry explicitly supplies Qiskit's
LEGACY_CUSTOM_INSTRUCTIONS on import. This corrects serialization only; ZX
rewrite/extraction and the scientific controls are unchanged. A small independent
Qiskit-unitary equivalence test remains mandatory before admitting the flow.
Both failed preflight directories and logs are retained.
