import argparse
import ast
import os
import pickle
from pathlib import Path

import numpy as np
from constants import CHAT_GPT_FOLDER
from datasets import get_df
from utils import get_logger, seed_everything, set_global_log_level

logger = get_logger(__name__)


def get_prompt(prompt_path):
    """
    Read a base-prompt from file.

    Args:
        prompt_path (str): Path to the baseprompt

    Returns:
        str: The base-prompt read.
    """
    prompt_path = Path(CHAT_GPT_FOLDER) / prompt_path
    with open(prompt_path, "r") as infile:
        prompt = infile.read()
    return prompt


def write_full_prompt(filepath, sentences, base_prompt):
    """
    Given sentences and base-prompt, write a full prompt with the sentences.

    Args:
        filepath (str): Path to where to save the prompt.
        sentences (list of str): List of the sentences / claims.
        base_prompt (str): The base-prompt. Use `get_prompt()`.
    """
    sub_size = len(sentences)
    with open(filepath, "w") as outfile:
        outfile.write(base_prompt)
        outfile.write("\n\n```\n")
        for i in range(sub_size):
            outfile.write(f"{i + 1}. {sentences.iloc[i]}\n")
        outfile.write("\n```\n\n")
        outfile.write("Please begin your analysis below:")


def draw_sample(dataset_type, sample_size=100, sub_sizes=[25, 50, 100], prompt_path="base_prompt.txt"):
    """
    Draws random claims. Saves the claims and the dataframe to file. Also loops over `sub_sizes`, and creates
    prompts with questions of suiting sizes. For instance, if `sample_size` is 100 and `25` is in `sub_sizes`,
    then it will create 4 prompts with the base-prompt and 25 of the 100 claims at a time.

    Args:
        dataset_type (str): Dataset type in ["train", "val", "test"]
        sample_size (int, optional): The total sample size. Defaults to 100.
        sub_sizes (list of int, optional): List of the sub-sizes to create prompts for. Defaults to [25, 50].
        prompt_path (str, optional): Path to the base-prompt. Defaults to "base_prompt.txt".

    Raises:
        ValueError: If `sample_size` is not a multiple of every element in `sub_sizes`.
    """
    # First draw the sample df and save all the questions and labels to file
    df = get_df(data_split=dataset_type, small=False)
    sample_df = df.sample(sample_size)
    sentences = sample_df["Sentence"]

    filename = "all_questions_" + str(sample_size) + ".txt"
    destination_path = Path(CHAT_GPT_FOLDER) / dataset_type / filename
    os.makedirs(Path(CHAT_GPT_FOLDER) / dataset_type, exist_ok=True)
    with open(destination_path, "w") as outfile:
        for i in range(args.sample_size):
            outfile.write(f"{i + 1}. {sentences.iloc[i]}\n")

    df_filename = "full_df_" + str(sample_size) + ".pkl"
    df_path = Path(CHAT_GPT_FOLDER) / dataset_type / df_filename
    with open(df_path, "wb") as outfile:
        pickle.dump(sample_df, outfile)

    # Now we have drawn the questions and saved, now we will make prompts for each sub-size.
    # We need many prompts for each sub-size that is less than the sample-size.
    base_prompt = get_prompt(prompt_path)

    for sub_size in sub_sizes:
        sub_size_foldername = "sub_" + str(sub_size)
        os.makedirs(Path(CHAT_GPT_FOLDER) / dataset_type / sub_size_foldername, exist_ok=True)
        if sample_size % sub_size != 0:
            raise ValueError(f"Argument `sample_size` must be multiple of sub-sizes. Was {sample_size=}, {sub_sizes=}")
        for i in range(int(sample_size / sub_size)):
            sub_df = sample_df.iloc[(i * sub_size): ((i + 1) * sub_size)]
            sentences = sub_df["Sentence"]
            filename = "prompt_sub" + str(sub_size) + "_n" + str(i + 1) + ".txt"
            file_path = Path(CHAT_GPT_FOLDER) / dataset_type / sub_size_foldername / filename
            write_full_prompt(file_path, sentences, base_prompt)


def read_answers(answers_path):
    """
    Reads ChatGPT answers and returns only the predictions.

    Args:
        answers_path (str): Path to the answers.

    Returns:
        np.array: Array of the predictions
    """
    with open(answers_path, "r") as infile:
        answers = infile.read()
    answers = ast.literal_eval(answers)
    predictions = [answer[1] for answer in answers]
    reasons = [answer[2] for answer in answers]
    predictions = np.array(predictions)
    return predictions, reasons


def evaluate_answers(dataset_type, sample_size=100, sub_sizes=[25, 50, 100], n_runs=1):
    """
    Evaluate ChatGPT answers against labels, for all subsizes in `sub_sizes`, for `n_runs` total of runs.
    `n_runs` marks how many time ChatGPT was ran.
    The answers must be saved as python lists on the correct filename: `answers_n<i>_r<j>.txt`, where <i> is the
    subset of answers (i is 1, 2, 3, 4 for `sub_sizes` equal to 25) and `<j>` is the run (if ChatGPT is asked more
    than once with the same answers).

    Args:
        dataset_type (str): Dataset type in ["train", "val", "test"]
        sample_size (int, optional): The sample size of the test-questions. Defaults to 100.
        sub_sizes (list, optional): The sub-sizes of questions to ask at a time. Defaults to [25, 50, 100].
        n_runs (int, optional): The amount of time ChatGPT was asked. Defaults to 1.

    Returns:
        list of np.array: List of the accuracies for each sub_size and each run.
    """
    sample_df_path = Path(CHAT_GPT_FOLDER) / dataset_type / ("full_df_" + str(sample_size) + ".pkl")
    sample_df = pickle.load(open(sample_df_path, "rb"))
    all_results = []

    for sub_size in sub_sizes:
        logger.info(f"Evaluating for {sub_size=}")
        sub_size_foldername = "sub_" + str(sub_size)
        total_accuracy = np.zeros(n_runs)
        for i in range(int(sample_size / sub_size)):
            sub_df = sample_df.iloc[(i * sub_size): ((i + 1) * sub_size)]
            labels = np.array([label[0] for label in sub_df["Label"]])
            for j in range(n_runs):
                answers_filename = "answers_n" + str(i + 1) + "_r" + str(j + 1) + ".txt"
                file_path = Path(CHAT_GPT_FOLDER) / dataset_type / sub_size_foldername / answers_filename
                predictions, reasons = read_answers(file_path)
                total_accuracy[j] += sum(predictions == labels) / sample_size
        logger.info(f"Results for {sub_size=}: {total_accuracy=}")
        all_results.append(total_accuracy)

    return all_results


if __name__ == "__main__":
    seed_everything(57)
    set_global_log_level("info")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type", choices=["train", "val", "test"], default="val", help="Data split to draw from")
    parser.add_argument("--sample_size", type=int, default=100, help="Amount of questions to ask for. ")
    parser.add_argument("--prompt_path", type=str, default="base_prompt.txt", help="Path / Filename to base prompt. ")
    parser.add_argument("--evaluate", action="store_true", help="Pass to evaluate after saving ChatGPT answers.")
    parser.add_argument("--n_runs", type=int, default=1, help="Amount of times ChatGPT was asked. ")

    args = parser.parse_args()

    seed_everything(57)
    sub_sizes = [25, 50, 100]

    if args.evaluate:
        logger.info("Evaluating answers...")
        all_results = evaluate_answers(
            dataset_type=args.dataset_type, sample_size=args.sample_size, sub_sizes=sub_sizes, n_runs=args.n_runs)
    else:
        logger.info("Drawing sample...")
        draw_sample(dataset_type=args.dataset_type, sample_size=args.sample_size,
                    sub_sizes=sub_sizes, prompt_path=args.prompt_path)
    logger.info("Done.")
