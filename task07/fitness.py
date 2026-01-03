from task07.ann import ANN


def evaluate_fitness(ann: ANN) -> float:
    """
    F(ANN) = (1/4) * Σ [1 - |(a XOR b) - mapped_output|]
    mapped_output is ANN output in [0,1].
    """
    test_cases = [
        (0, 0, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 0),
    ]

    total = 0.0
    for a, b, expected in test_cases:
        raw = ann.forward(float(a), float(b))  # [-1, 1]
        mapped = (raw + 1) / 2  # [0, 1]
        deviation = abs(expected - mapped)
        total += 1.0 - deviation

    return total / 4.0
