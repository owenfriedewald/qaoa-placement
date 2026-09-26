"""Alternative exact binary phase-separator architectures."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit

from placement_core import PlacementProblem, manhattan
from compact_occupant_circuit import binary_label_bits, binary_qubit
from binary_phase_separator import (
    append_binary_placement_phase_separator,
    append_register_label_match_phase,
    binary_data_qubits,
    cell_label_by_name,
)


def site_index_bits(problem: PlacementProblem) -> int:
    return math.ceil(math.log2(len(problem.sites)))


def default_site_codes(problem: PlacementProblem) -> tuple[int, ...]:
    return tuple(range(len(problem.sites)))


def gray_site_codes(problem: PlacementProblem) -> tuple[int, ...]:
    return tuple(idx ^ (idx >> 1) for idx in range(len(problem.sites)))


def min_weight_site_codes(problem: PlacementProblem) -> tuple[int, ...]:
    width = site_index_bits(problem)
    codes = sorted(range(2**width), key=lambda code: (code.bit_count(), code))
    return tuple(codes[: len(problem.sites)])


def location_qubit(problem: PlacementProblem, cell_idx: int, bit_idx: int, base: int | None = None) -> int:
    offset = binary_data_qubits(problem) if base is None else base
    return offset + cell_idx * site_index_bits(problem) + bit_idx


def shared_location_total_qubits(problem: PlacementProblem) -> int:
    return binary_data_qubits(problem) + len(problem.cells) * site_index_bits(problem)


def coordinate_bits(problem: PlacementProblem) -> tuple[int, int]:
    max_x = max(x for x, _y in problem.sites)
    max_y = max(y for _x, y in problem.sites)
    return max(1, math.ceil(math.log2(max_x + 1))), max(1, math.ceil(math.log2(max_y + 1)))


def coordinate_base(problem: PlacementProblem) -> int:
    return shared_location_total_qubits(problem)


def coordinate_total_qubits(problem: PlacementProblem) -> int:
    x_bits, y_bits = coordinate_bits(problem)
    return shared_location_total_qubits(problem) + len(problem.cells) * (x_bits + y_bits)


def coordinate_qubit(problem: PlacementProblem, cell_idx: int, axis: str, bit_idx: int) -> int:
    x_bits, y_bits = coordinate_bits(problem)
    per_cell = x_bits + y_bits
    base = coordinate_base(problem) + cell_idx * per_cell
    if axis == "x":
        return base + bit_idx
    if axis == "y":
        return base + x_bits + bit_idx
    raise ValueError(axis)


def _zero_pattern_flips(width: int, label: int) -> list[int]:
    return [bit for bit in range(width) if ((label >> bit) & 1) == 0]


def append_label_match_xor(
    circuit: QuantumCircuit,
    register: Sequence[int],
    label: int,
    target: int,
) -> None:
    """XOR target if register equals label, with no persistent flag ancilla."""

    width = len(register)
    flips = [register[bit] for bit in _zero_pattern_flips(width, label)]
    for qubit in flips:
        circuit.x(qubit)
    circuit.mcx(list(register), target)
    for qubit in reversed(flips):
        circuit.x(qubit)


def _append_cell_location_extraction(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    cell_idx: int,
    inverse: bool = False,
    site_codes: Sequence[int] | None = None,
) -> int:
    """Compute or uncompute one cell's occupied site index into scratch bits.

    On legal states exactly one site register contains ``cell_idx``.  The site
    index is XORed into the location register.  The same operation is its own
    inverse, but reversing the site/bit order preserves a clean circuit inverse
    for transpiler diagnostics.
    """

    occ_width = binary_label_bits(problem)
    loc_width = site_index_bits(problem)
    codes = tuple(site_codes) if site_codes is not None else default_site_codes(problem)
    ops: list[tuple[int, int]] = []
    for site_idx in range(len(problem.sites)):
        for bit_idx in range(loc_width):
            if (codes[site_idx] >> bit_idx) & 1:
                ops.append((site_idx, bit_idx))
    if inverse:
        ops = list(reversed(ops))
    count = 0
    for site_idx, bit_idx in ops:
        register = [binary_qubit(problem, site_idx, bit) for bit in range(occ_width)]
        target = location_qubit(problem, cell_idx, bit_idx)
        append_label_match_xor(circuit, register, cell_idx, target)
        count += 1
    return count


def append_shared_location_binary_phase_separator(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    gamma: float,
) -> dict[str, int]:
    """Append exact phase layer using shared cell-location extraction.

    The operation is exact on the legal binary occupant subspace.  Invalid
    states are not repaired; the extraction computes the XOR of all matching
    site indices for each real label and is uncomputed after the phase terms.
    """

    loc_width = site_index_bits(problem)
    extraction_ops = 0
    for cell_idx, _cell in enumerate(problem.cells):
        extraction_ops += _append_cell_location_extraction(circuit, problem, cell_idx, inverse=False)

    phase_terms = 0
    labels = cell_label_by_name(problem)
    for left_cell, right_cell, weight in problem.nets:
        left_idx = labels[left_cell]
        right_idx = labels[right_cell]
        left_register = [location_qubit(problem, left_idx, bit) for bit in range(loc_width)]
        right_register = [location_qubit(problem, right_idx, bit) for bit in range(loc_width)]
        for left_site_idx, left_site in enumerate(problem.sites):
            for right_site_idx, right_site in enumerate(problem.sites):
                distance = manhattan(left_site, right_site)
                if distance == 0:
                    continue
                append_register_label_match_phase(
                    circuit,
                    left_register,
                    left_site_idx,
                    right_register,
                    right_site_idx,
                    -float(gamma) * float(weight) * float(distance),
                )
                phase_terms += 1

    for cell_idx in reversed(range(len(problem.cells))):
        _append_cell_location_extraction(circuit, problem, cell_idx, inverse=True)

    return {
        "extraction_mcx_ops": extraction_ops * 2,
        "location_phase_terms": phase_terms,
        "location_bits": len(problem.cells) * loc_width,
    }


def walsh_z_coefficients(values: Sequence[float]) -> np.ndarray:
    """Return coefficients c_S for f(b)=sum_S c_S prod_i (-1)^b_i."""

    values_array = np.asarray(values, dtype=float)
    coeffs = values_array.copy()
    n = len(coeffs)
    step = 1
    while step < n:
        for start in range(0, n, 2 * step):
            for idx in range(start, start + step):
                a = coeffs[idx]
                b = coeffs[idx + step]
                coeffs[idx] = a + b
                coeffs[idx + step] = a - b
        step *= 2
    coeffs /= n
    return coeffs


def append_parity_phase(circuit: QuantumCircuit, qubits: Sequence[int], mask: int, coefficient: float) -> None:
    """Append exp(i coefficient prod Z_i) for selected qubits."""

    if abs(coefficient) < 1e-12:
        return
    selected = [qubits[idx] for idx in range(len(qubits)) if (mask >> idx) & 1]
    if not selected:
        circuit.global_phase += coefficient
        return
    target = selected[-1]
    controls = selected[:-1]
    for control in controls:
        circuit.cx(control, target)
    circuit.rz(-2.0 * coefficient, target)
    for control in reversed(controls):
        circuit.cx(control, target)


def parity_support(qubits: Sequence[int], mask: int) -> tuple[int, ...]:
    return tuple(qubits[idx] for idx in range(len(qubits)) if (mask >> idx) & 1)


def parity_term_records(
    problem: PlacementProblem,
    gamma: float,
    site_codes: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Return nonzero parity terms for the distance-lookup oracle.

    The returned terms are independent of ordering and are exact Walsh
    coefficients for the weighted Manhattan lookup tables.  Zero-support terms
    are represented as a separate global phase.
    """

    labels = cell_label_by_name(problem)
    records: list[dict[str, Any]] = []
    global_phase = 0.0
    term_id = 0
    for net_id, (left_cell, right_cell, weight) in enumerate(problem.nets):
        left_idx = labels[left_cell]
        right_idx = labels[right_cell]
        qubits = _location_register(problem, left_idx) + _location_register(problem, right_idx)
        coeffs = walsh_z_coefficients(distance_phase_values(problem, weight, gamma, site_codes=site_codes))
        for mask, coeff in enumerate(coeffs):
            coefficient = float(coeff)
            if abs(coefficient) < 1e-10:
                continue
            support = parity_support(qubits, mask)
            if not support:
                global_phase += coefficient
                continue
            records.append(
                {
                    "term_id": term_id,
                    "net_id": net_id,
                    "left_cell": left_cell,
                    "right_cell": right_cell,
                    "weight": float(weight),
                    "mask": int(mask),
                    "coefficient": coefficient,
                    "qubits": tuple(qubits),
                    "support": support,
                    "support_size": len(support),
                }
            )
            term_id += 1
    return records, float(global_phase)


def _support_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    return len(set(left["support"]) & set(right["support"]))


def _support_hamming(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    return len(set(left["support"]) ^ set(right["support"]))


def order_parity_terms(
    records: Sequence[Mapping[str, Any]],
    policy: str = "repository_order",
    beam_width: int = 4,
    candidate_pool: int = 10,
) -> list[Mapping[str, Any]]:
    """Order parity records without changing the term multiset."""

    if policy in ("repository_order", "grouped_by_net_then_mask"):
        return sorted(records, key=lambda term: (term["term_id"],))
    if policy == "greedy_support_overlap":
        if not records:
            return []
        remaining = {int(term["term_id"]): term for term in records}
        current = min(records, key=lambda term: (term["net_id"], term["mask"], term["term_id"]))
        ordered = [current]
        remaining.pop(int(current["term_id"]))
        while remaining:
            next_term = min(
                remaining.values(),
                key=lambda term: (
                    -_support_overlap(current, term),
                    _support_hamming(current, term),
                    term["support_size"],
                    term["mask"],
                    term["term_id"],
                ),
            )
            ordered.append(next_term)
            remaining.pop(int(next_term["term_id"]))
            current = next_term
        return ordered
    if policy == "beam_search":
        return _beam_order(records, beam_width=beam_width, candidate_pool=candidate_pool)
    raise ValueError(f"unknown parity ordering policy: {policy}")


def _beam_order(
    records: Sequence[Mapping[str, Any]],
    beam_width: int = 4,
    candidate_pool: int = 10,
) -> list[Mapping[str, Any]]:
    if not records:
        return []
    term_map = {int(term["term_id"]): term for term in records}
    starts = sorted(records, key=lambda term: (term["support_size"], term["mask"], term["term_id"]))[:beam_width]
    beam: list[tuple[list[int], set[int]]] = [([int(term["term_id"])], set(term_map) - {int(term["term_id"])}) for term in starts]
    while beam and beam[0][1]:
        next_beam: list[tuple[list[int], set[int]]] = []
        for prefix, remaining in beam:
            last = term_map[prefix[-1]]
            candidates = sorted(
                (term_map[term_id] for term_id in remaining),
                key=lambda term: (
                    -_support_overlap(last, term),
                    _support_hamming(last, term),
                    term["support_size"],
                    term["mask"],
                    term["term_id"],
                ),
            )[:candidate_pool]
            for term in candidates:
                term_id = int(term["term_id"])
                new_remaining = set(remaining)
                new_remaining.remove(term_id)
                next_beam.append((prefix + [term_id], new_remaining))
        next_beam.sort(key=lambda state: parity_order_metrics([term_map[term_id] for term_id in state[0]])["explicit_parity_cx"])
        beam = next_beam[:beam_width]
    best = min(beam, key=lambda state: parity_order_metrics([term_map[term_id] for term_id in state[0]])["explicit_parity_cx"])
    return [term_map[term_id] for term_id in best[0]]


def parity_order_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens: list[tuple[str, int, int] | tuple[str, int]] = []
    original = 0
    for term in records:
        support = tuple(int(q) for q in term["support"])
        if len(support) <= 1:
            tokens.append(("rz", support[-1] if support else -1))
            continue
        target = support[-1]
        controls = support[:-1]
        for control in controls:
            tokens.append(("cx", control, target))
            original += 1
        tokens.append(("rz", target))
        for control in reversed(controls):
            tokens.append(("cx", control, target))
            original += 1

    reduced: list[tuple[str, int, int] | tuple[str, int]] = []
    for token in tokens:
        if token[0] == "cx" and reduced and reduced[-1] == token:
            reduced.pop()
        else:
            reduced.append(token)

    loads: dict[int, int] = {}
    depths: dict[int, int] = {}
    explicit = 0
    for token in reduced:
        if token[0] != "cx":
            continue
        _kind, control, target = token
        explicit += 1
        layer = max(depths.get(control, 0), depths.get(target, 0)) + 1
        depths[control] = layer
        depths[target] = layer
        loads[control] = loads.get(control, 0) + 1
        loads[target] = loads.get(target, 0) + 1

    return {
        "original_parity_cx": original,
        "explicit_parity_cx": explicit,
        "adjacent_inverse_cancellations": original - explicit,
        "depth_proxy": max(depths.values(), default=0),
        "max_qubit_load": max(loads.values(), default=0),
    }


def append_ordered_parity_network(
    circuit: QuantumCircuit,
    records: Sequence[Mapping[str, Any]],
    cancel_adjacent_cx: bool = False,
) -> None:
    if not cancel_adjacent_cx:
        for term in records:
            append_parity_phase(circuit, term["qubits"], int(term["mask"]), float(term["coefficient"]))
        return
    tokens: list[tuple[str, int, int] | tuple[str, int, float]] = []
    for term in records:
        support = tuple(int(q) for q in term["support"])
        if not support:
            continue
        target = support[-1]
        angle = -2.0 * float(term["coefficient"])
        if len(support) == 1:
            tokens.append(("rz", target, angle))
            continue
        controls = support[:-1]
        for control in controls:
            _append_cancellable_cx(tokens, control, target)
        tokens.append(("rz", target, angle))
        for control in reversed(controls):
            _append_cancellable_cx(tokens, control, target)
    for token in tokens:
        if token[0] == "cx":
            _kind, control, target = token
            circuit.cx(control, target)
        else:
            _kind, target, angle = token
            circuit.rz(angle, target)


def _append_cancellable_cx(
    tokens: list[tuple[str, int, int] | tuple[str, int, float]],
    control: int,
    target: int,
) -> None:
    token: tuple[str, int, int] = ("cx", control, target)
    if tokens and tokens[-1] == token:
        tokens.pop()
    else:
        tokens.append(token)


def append_phase_polynomial_from_table(
    circuit: QuantumCircuit,
    qubits: Sequence[int],
    phase_values: Sequence[float],
) -> int:
    coeffs = walsh_z_coefficients(phase_values)
    terms = 0
    for mask, coeff in enumerate(coeffs):
        if abs(float(coeff)) < 1e-10:
            continue
        append_parity_phase(circuit, qubits, mask, float(coeff))
        terms += 1
    return terms


def _location_register(problem: PlacementProblem, cell_idx: int) -> list[int]:
    return [location_qubit(problem, cell_idx, bit) for bit in range(site_index_bits(problem))]


def distance_phase_values(
    problem: PlacementProblem,
    weight: float,
    gamma: float,
    site_codes: Sequence[int] | None = None,
) -> list[float]:
    width = site_index_bits(problem)
    dim = 2 ** (2 * width)
    codes = tuple(site_codes) if site_codes is not None else default_site_codes(problem)
    code_to_site = {code: site_idx for site_idx, code in enumerate(codes)}
    values = []
    for basis in range(dim):
        left = basis & ((1 << width) - 1)
        right = (basis >> width) & ((1 << width) - 1)
        if left in code_to_site and right in code_to_site:
            distance = manhattan(problem.sites[code_to_site[left]], problem.sites[code_to_site[right]])
        else:
            distance = 0
        values.append(-float(gamma) * float(weight) * float(distance))
    return values


def append_location_distance_phase_polynomial(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    gamma: float,
    ordering_policy: str = "repository_order",
    cancel_adjacent_cx: bool = False,
    site_codes: Sequence[int] | None = None,
) -> dict[str, int]:
    """Extract cell locations, then phase each net with a Walsh parity table."""

    extraction_ops = 0
    for cell_idx, _cell in enumerate(problem.cells):
        extraction_ops += _append_cell_location_extraction(circuit, problem, cell_idx, inverse=False, site_codes=site_codes)

    parity_records, global_phase = parity_term_records(problem, gamma, site_codes=site_codes)
    ordered_records = order_parity_terms(parity_records, ordering_policy)
    if global_phase:
        circuit.global_phase += global_phase
    append_ordered_parity_network(circuit, ordered_records, cancel_adjacent_cx=cancel_adjacent_cx)
    metrics = parity_order_metrics(ordered_records)

    for cell_idx in reversed(range(len(problem.cells))):
        _append_cell_location_extraction(circuit, problem, cell_idx, inverse=True, site_codes=site_codes)

    return {
        "extraction_mcx_ops": extraction_ops * 2,
        "parity_phase_terms": len(parity_records),
        "ordering_policy": ordering_policy,
        "cancel_adjacent_cx": int(cancel_adjacent_cx),
        **metrics,
        "location_bits": len(problem.cells) * site_index_bits(problem),
    }


def flag_base(problem: PlacementProblem) -> int:
    return shared_location_total_qubits(problem)


def flag_site_indices(site_codes: Sequence[int]) -> list[int]:
    return [site_idx for site_idx, code in enumerate(site_codes) if int(code).bit_count() > 0]


def flag_qubit(problem: PlacementProblem, cell_idx: int, site_idx: int, site_codes: Sequence[int]) -> int:
    nonzero_sites = flag_site_indices(site_codes)
    return flag_base(problem) + cell_idx * len(nonzero_sites) + nonzero_sites.index(site_idx)


def flag_reuse_total_qubits(problem: PlacementProblem, site_codes: Sequence[int] | None = None) -> int:
    codes = tuple(site_codes) if site_codes is not None else default_site_codes(problem)
    return shared_location_total_qubits(problem) + len(problem.cells) * len(flag_site_indices(codes))


def append_location_distance_phase_polynomial_flag_reuse(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    gamma: float,
    ordering_policy: str = "beam_search",
    cancel_adjacent_cx: bool = True,
    site_codes: Sequence[int] | None = None,
) -> dict[str, int]:
    """Extract cell locations with live equality flags, then apply parity phases.

    This computes each predicate ``site_register == cell_label`` once per
    cell/site, fans the flag into every set bit of the site code, and keeps
    flags live until the location registers are uncomputed.  The construction
    trades extra clean scratch qubits for fewer repeated MCX predicates.
    """

    codes = tuple(site_codes) if site_codes is not None else default_site_codes(problem)
    occ_width = binary_label_bits(problem)
    loc_width = site_index_bits(problem)
    nonzero_sites = flag_site_indices(codes)
    mcx_ops = 0
    fanout_cx = 0
    for cell_idx, _cell in enumerate(problem.cells):
        for site_idx in nonzero_sites:
            register = [binary_qubit(problem, site_idx, bit) for bit in range(occ_width)]
            target = flag_qubit(problem, cell_idx, site_idx, codes)
            append_label_match_xor(circuit, register, cell_idx, target)
            mcx_ops += 1
    for cell_idx, _cell in enumerate(problem.cells):
        for site_idx in nonzero_sites:
            flag = flag_qubit(problem, cell_idx, site_idx, codes)
            for bit_idx in range(loc_width):
                if (codes[site_idx] >> bit_idx) & 1:
                    circuit.cx(flag, location_qubit(problem, cell_idx, bit_idx))
                    fanout_cx += 1

    parity_records, global_phase = parity_term_records(problem, gamma, site_codes=codes)
    ordered_records = order_parity_terms(parity_records, ordering_policy)
    if global_phase:
        circuit.global_phase += global_phase
    append_ordered_parity_network(circuit, ordered_records, cancel_adjacent_cx=cancel_adjacent_cx)
    metrics = parity_order_metrics(ordered_records)

    for cell_idx in reversed(range(len(problem.cells))):
        for site_idx in reversed(nonzero_sites):
            flag = flag_qubit(problem, cell_idx, site_idx, codes)
            for bit_idx in reversed(range(loc_width)):
                if (codes[site_idx] >> bit_idx) & 1:
                    circuit.cx(flag, location_qubit(problem, cell_idx, bit_idx))
                    fanout_cx += 1
    for cell_idx in reversed(range(len(problem.cells))):
        for site_idx in reversed(nonzero_sites):
            register = [binary_qubit(problem, site_idx, bit) for bit in range(occ_width)]
            target = flag_qubit(problem, cell_idx, site_idx, codes)
            append_label_match_xor(circuit, register, cell_idx, target)
            mcx_ops += 1

    return {
        "extraction_mcx_ops": mcx_ops,
        "extraction_fanout_cx": fanout_cx,
        "flag_qubits": len(problem.cells) * len(nonzero_sites),
        "parity_phase_terms": len(parity_records),
        "ordering_policy": ordering_policy,
        "cancel_adjacent_cx": int(cancel_adjacent_cx),
        "site_codes": ",".join(str(code) for code in codes),
        **metrics,
        "location_bits": len(problem.cells) * site_index_bits(problem),
    }


def append_coordinate_lookup(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    inverse: bool = False,
) -> int:
    """Compute or uncompute x/y coordinate registers from extracted locations."""

    loc_width = site_index_bits(problem)
    x_bits, y_bits = coordinate_bits(problem)
    ops: list[tuple[int, int, str, int]] = []
    for cell_idx in range(len(problem.cells)):
        for site_idx, (x_coord, y_coord) in enumerate(problem.sites):
            for bit_idx in range(x_bits):
                if (x_coord >> bit_idx) & 1:
                    ops.append((cell_idx, site_idx, "x", bit_idx))
            for bit_idx in range(y_bits):
                if (y_coord >> bit_idx) & 1:
                    ops.append((cell_idx, site_idx, "y", bit_idx))
    if inverse:
        ops = list(reversed(ops))
    count = 0
    for cell_idx, site_idx, axis, bit_idx in ops:
        append_label_match_xor(
            circuit,
            _location_register(problem, cell_idx),
            site_idx,
            coordinate_qubit(problem, cell_idx, axis, bit_idx),
        )
        count += 1
    return count


def axis_distance_phase_values(bit_count: int, weight: float, gamma: float) -> list[float]:
    dim = 2 ** (2 * bit_count)
    values = []
    mask = (1 << bit_count) - 1
    for basis in range(dim):
        left = basis & mask
        right = (basis >> bit_count) & mask
        values.append(-float(gamma) * float(weight) * abs(left - right))
    return values


def append_coordinate_arithmetic_phase_polynomial(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    gamma: float,
) -> dict[str, int]:
    """Extract locations, look up coordinates, phase |dx|+|dy| by parity tables."""

    extraction_ops = 0
    for cell_idx, _cell in enumerate(problem.cells):
        extraction_ops += _append_cell_location_extraction(circuit, problem, cell_idx, inverse=False)
    coordinate_lookup_ops = append_coordinate_lookup(circuit, problem, inverse=False)

    labels = cell_label_by_name(problem)
    x_bits, y_bits = coordinate_bits(problem)
    parity_terms = 0
    for left_cell, right_cell, weight in problem.nets:
        left_idx = labels[left_cell]
        right_idx = labels[right_cell]
        x_qubits = [coordinate_qubit(problem, left_idx, "x", bit) for bit in range(x_bits)]
        x_qubits += [coordinate_qubit(problem, right_idx, "x", bit) for bit in range(x_bits)]
        y_qubits = [coordinate_qubit(problem, left_idx, "y", bit) for bit in range(y_bits)]
        y_qubits += [coordinate_qubit(problem, right_idx, "y", bit) for bit in range(y_bits)]
        parity_terms += append_phase_polynomial_from_table(
            circuit,
            x_qubits,
            axis_distance_phase_values(x_bits, weight, gamma),
        )
        parity_terms += append_phase_polynomial_from_table(
            circuit,
            y_qubits,
            axis_distance_phase_values(y_bits, weight, gamma),
        )

    append_coordinate_lookup(circuit, problem, inverse=True)
    for cell_idx in reversed(range(len(problem.cells))):
        _append_cell_location_extraction(circuit, problem, cell_idx, inverse=True)

    return {
        "extraction_mcx_ops": extraction_ops * 2,
        "coordinate_lookup_mcx_ops": coordinate_lookup_ops * 2,
        "parity_phase_terms": parity_terms,
        "location_bits": len(problem.cells) * site_index_bits(problem),
        "coordinate_bits": len(problem.cells) * sum(coordinate_bits(problem)),
    }


def candidate_a_phase_circuit(problem: PlacementProblem, gamma: float = 0.17) -> QuantumCircuit:
    circuit = QuantumCircuit(binary_data_qubits(problem) + 1)
    append_binary_placement_phase_separator(circuit, problem, gamma)
    return circuit


def shared_location_phase_circuit(problem: PlacementProblem, gamma: float = 0.17) -> QuantumCircuit:
    circuit = QuantumCircuit(shared_location_total_qubits(problem))
    append_shared_location_binary_phase_separator(circuit, problem, gamma)
    return circuit


def distance_lookup_phase_polynomial_circuit(
    problem: PlacementProblem,
    gamma: float = 0.17,
    ordering_policy: str = "repository_order",
    cancel_adjacent_cx: bool = False,
    site_codes: Sequence[int] | None = None,
) -> QuantumCircuit:
    circuit = QuantumCircuit(shared_location_total_qubits(problem))
    append_location_distance_phase_polynomial(
        circuit,
        problem,
        gamma,
        ordering_policy=ordering_policy,
        cancel_adjacent_cx=cancel_adjacent_cx,
        site_codes=site_codes,
    )
    return circuit


def distance_lookup_phase_polynomial_flag_reuse_circuit(
    problem: PlacementProblem,
    gamma: float = 0.17,
    ordering_policy: str = "beam_search",
    cancel_adjacent_cx: bool = True,
    site_codes: Sequence[int] | None = None,
) -> QuantumCircuit:
    codes = tuple(site_codes) if site_codes is not None else default_site_codes(problem)
    circuit = QuantumCircuit(flag_reuse_total_qubits(problem, codes))
    append_location_distance_phase_polynomial_flag_reuse(
        circuit,
        problem,
        gamma,
        ordering_policy=ordering_policy,
        cancel_adjacent_cx=cancel_adjacent_cx,
        site_codes=codes,
    )
    return circuit


def coordinate_arithmetic_phase_polynomial_circuit(problem: PlacementProblem, gamma: float = 0.17) -> QuantumCircuit:
    circuit = QuantumCircuit(coordinate_total_qubits(problem))
    append_coordinate_arithmetic_phase_polynomial(circuit, problem, gamma)
    return circuit
