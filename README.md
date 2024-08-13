# Fact or Fiction? Exploring Diverse Approaches to Fact Verification with Language Models

This repo contains the code and documentation to the article *Fact or Fiction? Exploring Diverse Approaches to Fact Verification with
Language Models*.

The paper assesses the performance of various language models at the [FactKG](https://arxiv.org/pdf/2104.06378) dataset. It tests a finetuned BERT, trains a [question answer graph neural network (QA-GNN)](https://arxiv.org/pdf/2104.06378) and prompts ChatGPT. The subgraph retrival method utilises simple logical methods, which are cheap to perform and resulted in improved performance. The QA-GNN trains efficiently due to frozen node and edge embeddings which can be computed in advance, and only once.

## Dependencies

The code requires `pandas`, `pytorch`, `numpy`, `transformers`, in addition to `pytorch_geometric` for QA-GNNs, and `nltk` and `spacy` for the contextulized subgraphs (not needed for single-step or direct subgraphs).

One has to run `python -m spacy download en_core_web_sm` to download the embeddings (it does not take long)

Here are the specific versions of the libraries used:

- **Numpy:** 1.25.2
- **Matplotlib:** 3.5.3
- **PyTorch:** 2.0.1
- **Pytorch Geometric**: 2.5.3
- **Pandas:** 2.2.2
- **Transformers:** 4.40.2
- **Scikit-learn:** 1.4.2
- **nltk:** 3.8.1
- **spacy** 3.7.4

and **Python** version 3.10.12.

### Installing Dependencies with Conda

- Run `conda env create -f environment.yaml` to create the conda environment.
- Run `conda activate fact_or_fiction_env` to activate the environment.

### Installing Dependencies with pip

- (Optional, but recommended) Create a virtual environment:
  - `python -m venv fact_or_fiction_venv`
  - Activate the environment:
    - On Windows: `fact_or_fiction_venv\Scripts\activate`
    - On macOS and Linux: `source fact_or_fiction_venv/bin/activate`
- Install the required packages: `pip install -r requirements.txt`

## Running the code

### Download the data

Please go [here](https://drive.google.com/drive/folders/1q0_MqBeGAp5_cBJCBf_1alYaYm14OeTk) to download the data. Use the `dbpedia_undirected_light.pickle` and `factkg.zip`. The light version was used in these experiments, so use it rather than the full knowledge graph in order to recreate the findings. The DBpedia knowledge graph is only used during preprocessing, not training or evaluating.

Unzip `factkg.zip` in `data/`, so that the `.pickle` files are saved in `data/factkg/`. Put the DBpedia `.pickle` file in `data/dbpedia/`. One can also change the paths and folder names in `constants.py`.

Chech the [FactKG paper](https://arxiv.org/pdf/2104.06378) for more information about the dataset.

Tip: Try setting `SMALL` to `True` in `glocal_settings.py` to test the runs really fast, since this index only a small part of the dataset.

Note: Because of the simple subgraph retrivals, the DBpedia knowledge graph is only used during preprocessing when finding the subgraphs. If the subgraphs are provided already, the DBpedia knowledge graph is not needed.

### Preprocess

If the subgraphs and embeddings file are already provided, the following steps can be skipped, and one can go directly to the training section. If the subgraphs are provided, but not the embedding file, one can skip step 1, but not 2. If none are provided, two preprocessing steps are necessary:

1. Retrieve the subgraphs (which is a non-trainable procedure that can be precomputed for each datapoint). Takes a couple of minutes, can be done on CPU. This uses the DBpedia knowledge graph.
2. For QA-GNN, precompute the embeddings for the nodes and the edges. Takes 20-60 minutes. Should only be done with CUDA. This uses the subgraphs found in the previous step.

For 1., one can use this script:

```cli
python retrieve_subgraphs.py --dataset_type all --method relevant
```

It takes a little while to load the knowledge graph (2-5 minutes), but the subgraph creations goes pretty fast. This will generate csv files locally in `./data/subgraphs/`. Changing `all` to `train`, `val` or `test` will only generate subgraphs for the respective split. The method `relevant` can be changed to `direct` or `one_hop`, for different retrival methods described in the paper. The `relevant` methods is the same as `contextualised` in the paper, and `one_hop` refers to `single-step`. If one wishes to create subgraphs for all three methods, one needs to run the line above three times.

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

Here are the specific script ran for the models in the paper (which includes the hyperparameters). Note that the first argument is simply the name of the model that will be saved, it can be set to anything.

BERT (single-step) (best-performing model) `python run_stuff.py bert_single_step --subgraph_type direct_filled --subgraph_to_use walkable --n_epochs 10 --batch_size 4 --learning_rate 0.000005`

QA-GNN (single-step): `python run_stuff.py qa_gnn_single_step --qa_gnn --subgraph_type relevant --n_epochs 30 --batch_size 128 --gnn_batch_norm --classifier_dropout 0.5 --gnn_dropout 0.1 --learning_rate 0.000001 --mix_graphs`

QA-GNN (contextual): `python run_stuff.py qa_gnn_contextual --qa_gnn --subgraph_type relevant --n_epochs 20 --batch_size 64 --gnn_batch_norm --classifier_dropout 0.5 --gnn_dropout 0.1 --learning_rate 0.000005`

QA-GNN (direct): `python run_stuff.py --model_name qa_gnn_direct --qa_gnn --subgraph_type direct --n_epochs 8 --batch_size 128 --gnn_batch_norm`.

BERT (contextual): `python run_stuff.py bert_contextual --subgraph_type relevant --subgraph_to_use connected --n_epochs 10 --batch_size 8 --learning_rate 0.000005`

BERT (direct): `python run_stuff.py bert_direct --subgraph_type direct --subgraph_to_use connected --n_epochs 10 --batch_size 4 --learning_rate 0.000005`

BERT (no subgraphs): `python run_stuff.py baseline_no_evidence --subgraph_type none --n_epochs 10 --batch_size 32`

### ChatGPT prompting

The code for running the ChatGPT experiments is in `run_chatgpt.py`, while text files for the prompts and answers can be found in `chatgpt/`.

If one wish to create a new prompt with new questions, run:

```cli
python run_chatgpt.py --dataset_type val --sample_size 100 --prompt_path base_prompt.txt
```

where you can change `val` to `train` or `test` to draw from the respective split, and `base_prompt.txt` can be changed to another file in `chatgpt/` that contains the base prompt.

The prompt with questions created was manually copy pasted into ChatGPT 4's website. The answers were manually copy pasted into answer text files. The folder contains the answers and questions for three runs of 100 drawn test questions, both with a split of 20, 50 and 100 questions at a time. To evaluate everything, run:

```cli
python run_chatgpt.py --dataset_type test --evaluate --n_runs 3
```

### Thank You

![no gif :(](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHdlZzhnOXJtaGp1ZG1vOHpudWtkaTExdTM3Ync5OHYxNmw5dGg0diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/mcsPU3SkKrYDdW3aAU/giphy.gif)
