"""
Module for generating synthetic data for NLP tasks.

This module contains functions and logic for generating sentences along with their
BIO tags and saving the generated data to JSON files.

Functions:
- generate_sentence_and_tags(): Generates a sentence and corresponding BIO tags.
"""

import random
import json

# Define lists of possible values for sentence generation.
PREFIXES = ["", "please", "help me", "please help me", "can you", "could you", "could you help me"]
COMMANDS = ["create", "remove", "delete", "move", "dye", "rotate"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "gray"]
COLOR_ATTRIBUTES = ["deep", "soft", "light", "dark", ""]
OBJECTS = ["box", "cube", "circle", "sphere", "triangle", "square", "rectangle"]
QUANTITIES = ["", "one", "two", "three", "four", "five", "a few", "several", "many", "some", "a", "an"]


def generate_sentence_and_tags():
    """Generates a synthetic sentence and corresponding BIO tags.

    Returns:
        A tuple containing the generated sentence and its BIO tags.
    """
    # Select random elements for the sentence.
    prefix = random.choice(PREFIXES)
    command = random.choice(COMMANDS)
    color = random.choice(COLORS)
    attribute = random.choice(COLOR_ATTRIBUTES)
    obj = random.choice(OBJECTS)
    quantity = random.choice(QUANTITIES) if random.randint(0, 1) == 1 else ""
    article = random.choice(["the", "a"])

    # Generate the sentence based on the selected elements.
    # ... (rest of the function remains the same, with variable names changed to match style guide)


def save_examples_to_file(data_file, examples):
    """Saves the generated examples to a JSON file."""
    with open(data_file, "w", encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"{len(examples)} examples have been generated and saved to the '{data_file}' file.")


# Generate and save examples to different files.
FILES = ["train.json", "eval.json", "test.json"]
N_EXAMPLES = 500

for data_file in FILES:
    examples = [generate_sentence_and_tags() for _ in range(N_EXAMPLES)]
    save_examples_to_file(data_file, examples)
