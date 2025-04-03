import numpy as np
import random


def uniform_samples(range_start, range_end, num_samples=50):
    """均匀采样"""
    step_size = (range_end - range_start + 1) / num_samples
    samples = []

    for i in range(num_samples):
        part_start = range_start + int(i * step_size)
        part_end = range_start + int((i + 1) * step_size) - 1

        if part_end > range_end:
            part_end = range_end
        if part_start > part_end:
            part_end = part_start

        sample = np.random.randint(part_start, part_end + 1)
        samples.append(sample)

    np.random.shuffle(samples)
    return samples


def uniform_power_of_two(range_start, range_end, num_samples=3000):
    """二次幂均匀采样"""
    if range_end < range_start or num_samples <= 0:
        return []

    powers_of_two = [2 ** i for i in range(int(np.log2(range_start)), int(np.log2(range_end)) + 1)
                     if 2 ** i >= range_start and 2 ** i <= range_end]

    step_size = num_samples // len(powers_of_two)
    samples = step_size * powers_of_two
    samples2 = random.sample(powers_of_two, num_samples - step_size * len(powers_of_two))
    samples = samples + samples2
    np.random.shuffle(samples)
    return samples