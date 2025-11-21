import random
import string

TARGET = "charles darwin was always seasick"
CHARS = string.ascii_lowercase + " "


def random_string(length):
    """Generate a random string of given length from lowercase letters and spaces."""
    return "".join(random.choice(CHARS) for _ in range(length))


def fitness(s):
    """Calculate fitness as number of correct characters in correct positions."""
    return sum(1 for i, char in enumerate(s) if char == TARGET[i])


def mutate(s):
    """Mutate string by randomly changing one character."""
    s_list = list(s)
    pos = random.randint(0, len(s) - 1)
    s_list[pos] = random.choice(CHARS)
    return "".join(s_list)


def evolve(print_every=1):
    """Run the hill climber evolution algorithm."""
    current = random_string(len(TARGET))
    current_fitness = fitness(current)
    generation = 0

    print(f"Target:  {TARGET}")
    print(f"Gen {generation:5d}: {current} (fitness: {current_fitness})")

    while current_fitness < len(TARGET):
        generation += 1
        candidate = mutate(current)
        candidate_fitness = fitness(candidate)

        # Accept if fitness doesn't decrease
        if candidate_fitness >= current_fitness:
            current = candidate
            current_fitness = candidate_fitness

        if generation % print_every == 0 or current_fitness == len(TARGET):
            print(f"Gen {generation:5d}: {current} (fitness: {current_fitness})")

    return generation


def run_experiments(n_runs=100):
    """Run multiple experiments and calculate average generations needed."""
    print(f"\nRunning {n_runs} experiments...")
    generations_list = []

    for i in range(n_runs):
        # Run without printing each generation
        current = random_string(len(TARGET))
        current_fitness = fitness(current)
        generation = 0

        while current_fitness < len(TARGET):
            generation += 1
            candidate = mutate(current)
            candidate_fitness = fitness(candidate)

            if candidate_fitness >= current_fitness:
                current = candidate
                current_fitness = candidate_fitness

        generations_list.append(generation)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_runs} runs")

    avg = sum(generations_list) / len(generations_list)
    min_gen = min(generations_list)
    max_gen = max(generations_list)

    print(f"\nResults from {n_runs} runs:")
    print(f"  Average generations: {avg:.2f}")
    print(f"  Min generations: {min_gen}")
    print(f"  Max generations: {max_gen}")

    return generations_list


if __name__ == "__main__":
    print("=" * 70)
    print("Task 3.1: Single run with output")
    print("=" * 70)
    generations = evolve(print_every=100)
    print(f"\nCompleted in {generations} generations")

    print("\n" + "=" * 70)
    print("Task 3.3: Empirical analysis")
    print("=" * 70)
    run_experiments(100)
