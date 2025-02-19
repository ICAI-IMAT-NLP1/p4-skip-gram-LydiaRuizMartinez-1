from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch
import torch.nn.functional as F
import numpy as np
import random

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file

    # Preprocess and tokenize the text
    # TODO
    tokens: List[str] = tokenize(text)

    return tokens


def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # TODO
    # Count word frequencies
    word_counts: Counter = Counter(words)

    # Sorting the words from most to least frequent
    sorted_vocab: List[str] = sorted(word_counts, key=word_counts.get, reverse=True)

    # Create vocab_to_int and int_to_vocab dictionaries
    vocab_to_int: Dict[str, int] = {word: idx for idx, word in enumerate(sorted_vocab)}
    int_to_vocab: Dict[int, str] = {idx: word for word, idx in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def subsample_words(
    words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5
) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the
    presence of frequent words according to Mikolov's subsampling technique. This method
    calculates the probability of keeping each word in the dataset based on its frequency,
    with more frequent words having a higher chance of being discarded. The process helps
    in balancing the word distribution, potentially leading to faster training and better
    representations by focusing more on less frequent words.

    Args:
        words (list): List of words to be subsampled.
        vocab_to_int (dict): Dictionary mapping words to unique integers.
        threshold (float): Threshold parameter controlling the extent of subsampling.


    Returns:
        List[int]: A list of integers representing the subsampled words, where some high-frequency words may be removed.
        Dict[str, float]: Dictionary associating each word with its frequency.
    """
    # TODO
    # Convert words to integers
    word_counts: Dict[str, int] = dict(Counter(words))
    total_words: int = len(words)

    int_words: List[int] = [
        vocab_to_int[word] for word in words if word in vocab_to_int
    ]

    # Calculate frequency of each word
    word_counts = {word: words.count(word) for word in set(words)}
    total_words = len(words)
    freqs: Dict[str, float] = {
        word: count / total_words for word, count in word_counts.items()
    }

    # Compute the subsampling probability
    discard_probs = {
        word: 1 - np.sqrt(threshold / freq) for word, freq in freqs.items()
    }

    # Subsample words
    train_words: List[int] = [
        word
        for word in int_words
        if np.random.random() > discard_probs.get(words[word], 0)
    ]

    return train_words, freqs


def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """
    # TODO
    # Randomly select a window size from 1 to window_size
    random_window: int = random.randint(1, window_size)

    # Define the range for context words
    start: int = max(0, idx - random_window)
    end: int = min(len(words), idx + random_window + 1)

    # Collect words around idx, excluding the target word itself
    target_words: List[int] = [words[i] for i in range(start, end) if i != idx]

    return target_words


def get_batches(
    words: List[int], batch_size: int, window_size: int = 5
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """
    # TODO
    # Iterate over the dataset in increments of batch_size
    for idx in range(0, len(words), batch_size):
        inputs: List[int] = list()
        targets: List[int] = list()

        # Define the upper limit of the batch
        batch_limit: int = min(idx + batch_size, len(words))

        # Iterate over the words in the current batch range
        for i in range(idx, batch_limit):
            # Get context words surrounding the target word at index i
            target_words = get_target(words, i, window_size)
            # Add the center word multiple times, once for each context word
            inputs.extend([words[i]] * len(target_words))
            # Add corresponding context words
            targets.extend(target_words)

        yield inputs, targets


def cosine_similarity(
    embedding: torch.nn.Embedding,
    valid_size: int = 16,
    valid_window: int = 100,
    device: str = "cpu",
):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    # TODO
    # Select valid_size random word indices within valid_window range
    valid_examples: torch.Tensor = (
        torch.randint(0, valid_window, (valid_size,), device=device).clone().detach()
    )

    # Extract embeddings of selected words
    valid_embeddings: torch.Tensor = embedding(
        valid_examples
    )  # Shape: (valid_size, embedding_dim)

    # Normalize the embeddings to unit vectors
    norm_valid_embeddings: torch.Tensor = F.normalize(
        valid_embeddings, p=2, dim=1
    )  # Shape: (valid_size, embedding_dim)
    norm_embedding_matrix: torch.Tensor = F.normalize(
        embedding.weight, p=2, dim=1
    )  # Shape: (vocab_size, embedding_dim)

    # Compute cosine similarity: (valid_embeddings @ embedding_matrix.T)
    similarities: torch.Tensor = torch.matmul(
        norm_valid_embeddings, norm_embedding_matrix.T
    )  # Shape: (valid_size, vocab_size)

    return valid_examples, similarities
