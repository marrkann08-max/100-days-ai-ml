import numpy as np

INPUT_LABELS = ("level", "fair share", "food left", "energy", "month left")
OUTPUT_LABELS = ("portion",)

N_INPUTS = len(INPUT_LABELS)
N_HIDDEN = 6
N_OUTPUTS = len(OUTPUT_LABELS)

# Each genome is one flat row: input weights, hidden biases, output weights, output biases.
SHAPES = ((N_INPUTS, N_HIDDEN), (N_HIDDEN,), (N_HIDDEN, N_OUTPUTS), (N_OUTPUTS,))
GENOME_SIZE = sum(int(np.prod(shape)) for shape in SHAPES)


def random_genomes(rng, *shape):
    return rng.normal(0, .6, (*shape, GENOME_SIZE))


def unpack(genomes):
    genomes = np.atleast_2d(genomes)
    parts, start = [], 0
    for shape in SHAPES:
        size = int(np.prod(shape))
        parts.append(genomes[:, start:start + size].reshape(len(genomes), *shape))
        start += size
    return parts


def think(genomes, inputs):
    """Run every brain on its own inputs. Returns hidden and output activations."""
    w1, b1, w2, b2 = unpack(genomes)
    hidden = np.tanh(np.einsum("ni,nih->nh", inputs, w1) + b1)
    outputs = np.tanh(np.einsum("nh,nho->no", hidden, w2) + b2)
    return hidden, outputs
