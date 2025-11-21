# Task 3: Analysis

## Task 3.2: Why is this a "good-natured" optimization problem?

This is a good-natured optimization problem for several reasons:

1. **No local maxima**: The fitness landscape has no local maxima except the global maximum. Every incorrect character can be fixed independently without breaking other correct characters. Once a character is correct, we never accept a mutation that makes it incorrect.

2. **Monotonic improvement**: The hill climber never backtracks (fitness can only stay the same or improve), which works perfectly here because there are no fitness valleys to escape from.

3. **Independent dimensions**: Each character position is essentially an independent optimization problem. Fixing one character doesn't make it harder to fix another.

4. **Linear fitness landscape**: The fitness function is additive - each correct character contributes +1 to fitness regardless of other characters.

## Task 3.3: Mathematical estimation

### Analysis:

At any generation with fitness $f$ (where $0 \leq f < 33$), there are:
- $f$ correct characters (we don't want to change these)
- $33 - f$ incorrect characters (we want to fix these)

When we mutate:
1. Probability of selecting an incorrect character: $\frac{33-f}{33}$
2. Given we selected an incorrect character, probability of changing it to the correct character: $\frac{1}{27}$ (27 possible characters: 26 letters + space)
3. Probability of improvement: $\frac{33-f}{33} \times \frac{1}{27} = \frac{33-f}{891}$

The expected number of generations to improve from fitness $f$ to $f+1$ is:
$$E[G_f] = \frac{891}{33-f}$$

Total expected generations to go from fitness 0 to 33:
$$E[G_{total}] = \sum_{f=0}^{32} \frac{891}{33-f} = 891 \sum_{k=1}^{33} \frac{1}{k} = 891 \times H_{33}$$

where $H_{33}$ is the 33rd harmonic number.

$$H_{33} = \sum_{k=1}^{33} \frac{1}{k} \approx 4.027$$

Therefore:
$$E[G_{total}] \approx 891 \times 4.027 \approx 3,588 \text{ generations}$$

### Note on initial fitness:

The expected initial fitness is approximately $\frac{33}{27} \approx 1.22$, so we typically don't start from fitness 0. A more refined estimate would account for this, giving us roughly:

$$E[G_{total}] \approx 891 \times (H_{33} - H_1) \approx 891 \times 3.027 \approx 2,697 \text{ generations}$$