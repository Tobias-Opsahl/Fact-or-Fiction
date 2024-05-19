import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from constants import CHAT_GPT_FOLDER
from datasets import get_df
from evaluate import evaluate_chatgpt
from utils import get_logger, seed_everything, set_global_log_level

logger = get_logger(__name__)


def get_prompt(prompt_path):
    prompt_path = Path(CHAT_GPT_FOLDER) / prompt_path
    with open(prompt_path, "r") as infile:
        prompt = infile.read()
    return prompt


def write_sample(args):
    df = get_df(data_split=args.dataset_type)
    sample_df = df.sample(args.sample_size)
    prompt = get_prompt(args.prompt_path)
    sentences = sample_df["Sentence"]

    destination_path = Path(CHAT_GPT_FOLDER) / args.destination_path
    with open(destination_path, "w") as outfile:
        outfile.write(prompt)
        outfile.write("\n\n```\n")
        for i in range(args.sample_size):
            outfile.write(f"{i + 1}. {sentences.iloc[i]}\n")
        outfile.write("\n```\n\n")
        outfile.write("Please begin your analysis below:")

    df_path = Path(CHAT_GPT_FOLDER) / (args.destination_path.replace(".txt", "") + "_df.pkl")
    with open(df_path, "wb") as outfile:
        pickle.dump(sample_df, outfile)


def read_df(labels_path):
    labels_path = Path(CHAT_GPT_FOLDER) / labels_path
    df = pd.read_pickle(labels_path)
    return df


def read_answers(answers_path):
    answers_path = Path(CHAT_GPT_FOLDER) / answers_path
    with open(answers_path, "r") as infile:
        answers = infile.read()
    answers = ast.literal_eval(answers)
    answers = [answer[1] for answer in answers]
    answers = np.array(answers)
    return answers


def evaluate(answers_path, labels_path):
    answers = read_answers(answers_path)
    df = read_df(labels_path)
    labels = df["Label"].values
    accuracy = sum(answers == labels) / len(labels)
    logger.info(f"Accuracy: {accuracy:.2f}. ")
    return accuracy


def evaluate_tests(n_questions_eval=20):
    if n_questions_eval == 20:
        test_predictions_filenames = [
            "test_answers_q20_1.txt", "test_answers_q20_2.txt", "test_answers_q20_3.txt",
            "test_answers_q20_4.txt", "test_answers_q20_5.txt"]
        test_df_filenames = [
            "test_q20_1_df.pkl", "test_q20_2_df.pkl", "test_q20_3_df.pkl",
            "test_q20_4_df.pkl", "test_q20_5_df.pkl"]
    elif n_questions_eval == 50:
        test_predictions_filenames = ["test_answers_q50_1.txt", "test_answers_q50_2.txt"]
        test_df_filenames = ["test_q50_1_df.pkl", "test_q50_2_df.pkl"]

    test_dict = {"predictions": [], "labels": [], "metadata": []}
    for answers_file, df_file in zip(test_predictions_filenames, test_df_filenames):
        answers = read_answers(answers_file)
        df = read_df(df_file)
        test_dict["predictions"].extend(list(answers))
        test_dict["labels"].extend(list(df["Label"]))
        test_dict["metadata"].extend(list(df["Metatada"]))

    df = pd.DataFrame(test_dict)

    evaluate_chatgpt(df)


if __name__ == "__main__":
    seed_everything(57)
    set_global_log_level("info")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type", choices=["train", "val", "test"], default="val", help="Data split to draw from")
    parser.add_argument("--sample_size", type=int, default=20, help="Amount of questions to ask for. ")
    parser.add_argument("--prompt_path", type=str, default="base_prompt1.txt", help="Path / Filename to base prompt. ")
    parser.add_argument("--destination_path", type=str, default="sample1.txt", help="Name of output file.")
    parser.add_argument(
        "--answers_path", type=str, default="", help="Include filename to ChatGPT to evaluate on single prompt")
    parser.add_argument(
        "--labels_path", type=str, default="", help="Include filename to df with labels to evaluate on single prompt.")
    parser.add_argument("--seed_offset", type=int, default=0, help="Offset seed to create different samples.")
    parser.add_argument("--evaluate_tests", action="store_true", help="Pass to evaluate on hardcoded test files.")
    parser.add_argument(
        "--n_questions_eval", type=int, choices=[20, 50], default=20, help="Number of questions to evaluate for")
    args = parser.parse_args()

    seed_everything(57 + args.sample_size + args.seed_offset)  # Allow for different sample of same size

    if args.evaluate_tests:
        logger.info("Evaluating test results:")
        evaluate_tests(args.n_questions_eval)
    elif args.answers_path != "" and args.labels_path != "":
        logger.info("Beggining evaluation:")
        evaluate(args.answers_path, args.labels_path)
    else:
        logger.info("Beginning sampling and prompt creation:")
        write_sample(args)
    logger.info("Done.")
