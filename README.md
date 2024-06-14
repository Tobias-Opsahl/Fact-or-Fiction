# Fact or Fiction? Exploring Diverse Approaches to Fact Verification with Language Models

This repo contains the code and documentation to the article *Fact or Fiction? Exploring Diverse Approaches to Fact Verification with
Language Models*.

The paper assesses the performance of various language models at the [FactKG](https://arxiv.org/pdf/2104.06378) dataset. It tests a finetuned BERT, trains a [question answer graph neural network (QA-GNN)](https://arxiv.org/pdf/2104.06378) and prompts ChatGPT. The subgraph retrival method utilises simple logical methods, which are cheap to perform and resulted in improved performance. The QA-GNN trains efficiently due to frozen node and edge embeddings which can be computed in advance, and only once.

## Requirements

The code requires `pandas`, `pytorch`, `numpy`, `transformers`, in addition to `pytorch_geometric` for QA-GNNs, and `nltk` and `spacy` for the contextulized embeddings. Additionally, one have to run `python -m spacy download en_core_web_sm` to download the embeddings (it does not take long).

## Running the code

Before training the models, one needs to run some preprocessing of the datasets. The `.csv` file from the FactKG dataset should be downloaded and put in `./data/` (or change the path in `constants.py`). One also needs the `DBpedia` knowledge graph (for preprocessing only, not training). The FactKG dataset provides a light version of DBpedia that is recommened to use. All the paths are saved in `constants.py`, so please make sure that these corresponds to your paths.

Chech the [FactKG paper](https://arxiv.org/pdf/2104.06378) to download the data.

Tip: Try setting `SMALL` to `True` in `glocal_settings.py` to test the runs really fast, since this index only a small part of the dataset.

### Preprocess

Two preprocessing steps are necessary:

1. Retrieve the subgraphs (which is a non-trainable procedure that can be precomputed for each datapoint). Takes a couple of minutes, can be done on CPU.
2. For QA-GNN, precompute the embeddings for the nodes and the edges. Takes 20-60 minutes. Should only be done with CUDA.

Both of them needs the `DBpedia` knowledge graph, which takes a couple mintues to load. After preprocessing, the knowledge graph is not needed anymore (for training and evaluation).

For 1., one can use this script:

```cli
python retrieve_subgraphs.py --dataset_type all --method relevant
```

This will generate csv files locally in `./data/subgraphs/`. Changing `all` to `train`, `val` or `test` will only generate subgraphs for the respective split. The method `relevant` can be changed to `direct` or `one_hop`, for different retrival methods described in the paper. The `relevant` methods is the same as `contextualised` in the paper, and `one_hop` refers to `single-step`. If one wishes to create subgraphs for all three methods, one needs to run the line above three times.

For 2., one can run:

```cli
python make_subgraph_embeddings.py --dataset_type all --subgraph_type relevant --batch_size 64
```

Again, one can change `all` to only do a certain split, and change `relevant` to some other method. One will need to run the line three times to generate embeddings for all the methods, but words that are already embedded will not be embedded again, so it is significantly faster the second and third time it is ran. It will create a pickle file `./data/embeddings.pkl` with a python dictionary of the embeddings.

### Training models

Once the preprocessing is done, one can train the models. The can be done with:

```cli
python run_stuff.py model_name --subgraph_type relevant
```

This will both train and evaluate the model, and save the training and evaluation results in `results/model_name_history.pkl`. All of the folders can be changed in `constants.py` if one wishes.

Including the argument ``--qa_gnn`` will train a QAGNN, if not, a BERT is finetuned. There are lots of arguments to change, which should be well documented in the argparse module (run `python run_stuff.py -h`).

When a model is trained, it will be automatically saved in `./models/`. A trained model can be evaluated only by passing `--evaluate_only` and `--state_dict_path models/model_name.pth`.

Here are the specific script ran for the models in the paper (which includes the hyperparameters):

BERT (single-step) (best-performing model) `python run_stuff.py baseline_evidence6 --subgraph_type direct_filled --subgraph_to_use walkable --n_epochs 10 --batch_size 4 --learning_rate 0.000005`

QA-GNN (single-step): `CUDA_VISIBLE_DEVICES=2 nohup python run_stuff.py qa_gnn44 --qa_gnn --subgraph_type relevant --n_epochs 30 --batch_size 128 --gnn_batch_norm --classifier_dropout 0.5 --gnn_dropout 0.1 --learning_rate 0.000001 --mix_graphs`

QA-GNN (contextual): `python run_stuff.py qa_gnn33 --qa_gnn --subgraph_type relevant --n_epochs 20 --batch_size 64 --gnn_batch_norm --classifier_dropout 0.5 --gnn_dropout 0.1 --learning_rate 0.000005`

QA-GNN (direct): `python run_stuff.py --model_name qa_gnn15 --qa_gnn --subgraph_type direct --n_epochs 8 --batch_size 128 --gnn_batch_norm`.

BERT (contextual): `python run_stuff.py baseline_evidence3 --subgraph_type relevant --subgraph_to_use connected --n_epochs 10 --batch_size 8 --learning_rate 0.000005`

BERT (direct): `python run_stuff.py baseline_evidence7 --subgraph_type direct --subgraph_to_use connected --n_epochs 10 --batch_size 4 --learning_rate 0.000005`

BERT (no subgraphs): `python run_stuff.py baseline_no_evidence --subgraph_type none --n_epochs 10 --batch_size 32`

### ChatGPT prompting

There are code aviable in `make_chatgpt_prompt.py` to create the ChatGPT prompts easily, text files for the prompts and answers can be found in `chatgpt/`. Run:

```cli
python make_chatgpt_prompt.py --dataset_type val --sample_size 20 --prompt_path base_prompt2.txt --destination_path sample.txt --seed_offset 0
```

This will add 20 validation questions to the prompt found in `base_prompt.txt`, saved in `sample.txt`. It will also save the rest of the dataframe (including the labels) as `sample_df.pkl`. If one wishes to have different samples of the same size, use different numbers of `--seed_offset`. If one wishes to change the prompt, make changes and send the prompt's path as `--prompt_path`.

The prompt (as in `sample.txt`) was manually copy pasted into ChatGPT 4's website. The answers were manually copy pasted into answer text files. To evaluate an answer (after saving the answers), run:

```cli
python make_chatgpt_prompt.py --answers_path test_answers_q20_1.txt --labels_path test_answers_q20_df.pkl
```

Changing the paths to how the answers are saved and what the dataframe was saved as. The reading assumes all files lie in `chatgpt/`.

Both the test prompts and answers used in the paper are provided in `chatgpt/`. To evaluate on all of it, run:

```cli
python make_chatgpt_prompt.py --evaluate_tests --n_questions_eval 20
```

Where 20 can be changed to 50 for the 50 question version.

### Thank You

![no gif :(](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHdlZzhnOXJtaGp1ZG1vOHpudWtkaTExdTM3Ync5OHYxNmw5dGg0diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/mcsPU3SkKrYDdW3aAU/giphy.gif)
