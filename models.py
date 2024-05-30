import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoModelForSequenceClassification, AutoModel
from utils import get_logger

logger = get_logger(__name__)


def get_bert_model(model_name="bert", include_classifier=True, num_labels=2, freeze_base_model=False,
                   freeze_up_to_pooler=True, dropout_rate=0, use_roberta=False):
    """
    Load a pretrained BERT model with desired configurations.

    Args:
        model_name (str, optional): Name of the model, will be saved as a class variable. Defaults to "bert".
        include_classifier (bool, optional): Include a classification layer. Defaults to True.
        num_labels (int, optional): Number of outputs if classification layer is included. Defaults to 2.
        freeze_base_model (bool, optional): Will freeze all layers up to the classification layer. Defaults to False.
        freeze_up_to_pooler (bool, optional): Will freeze all layers until the last layer in BERT, the pooler,
            with approximately 500k parameters. Defaults to True.
        dropout_rate (int, optional): Dropout rate for the classification layer. Defaults to 0.
        use_roberta (bool): If `True`, will use RoBERTa instead of BERT.

    Returns:
        transformer model: The loaded model.
    """
    if freeze_base_model and freeze_up_to_pooler:
        logger.warn("Both `freeze_base_model` and `freeze_up_to_pooler` is True. Freezing base model.")

    if use_roberta:
        model_name = "roberta-base"
    else:
        model_name = "bert-base-uncased"

    if include_classifier:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir="./cache", trust_remote_code=True, num_labels=num_labels,
            output_hidden_states=True
        )
    else:
        model = AutoModel.from_pretrained(model_name)

    model.name = model_name
    if freeze_base_model:
        for params in model.base_model.parameters():
            params.requires_grad = False
    elif freeze_up_to_pooler:
        for name, params in model.base_model.named_parameters():
            if not name.startswith("pooler"):
                params.requires_grad = False

    if dropout_rate != 0:
        classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.classifier.in_features, model.classifier.out_features)
        )
        model.classifier = classifier

    return model


class QAGNN(nn.Module):
    """
    Implementation of Quastion Answer Graph Neural Network model
    """
    def __init__(self, model_name, n_gnn_layers=2, gnn_hidden_dim=256, gnn_out_features=256, lm_layer_features=None,
                 gnn_batch_norm=True, freeze_base_model=False, freeze_up_to_pooler=True, gnn_dropout=0.3,
                 classifier_dropout=0.2, lm_layer_dropout=0.4, use_roberta=False):
        """
        Args:
            model_name (str): Name of the model, will be saved as a class variable.
            n_gnn_layers (int, optional): Number of layers in the GNN. Defaults to 2.
            gnn_hidden_dim (int, optional): Number of nodes in GNN layers. Defaults to 256.
            gnn_out_features (int, optional): Number of output nodes from the GNN. Defaults to 256.
            lm_layer_features (int): If not `None`, will add a linear layer after the claim embedding that will be
                used in the classification layer, concatenated with the GNN output. The layer will have
                `lm_layer_features` nodes, and a dropout of `lm_layer_dropout`. If `None`, will use the lm (bert)
                embedding concatenated with the GNN output for the classification layer.
            gnn_batch_norm (bool, optional): Whether or not to apply batch norm between the GNN layers.
                Defaults to True.
            freeze_base_model (bool, optional): Freeze the base model of the Bert language model. Defaults to False.
            freeze_up_to_pooler (bool, optional): Freeze up to the last part of the Bert model. Defaults to True.
            gnn_dropout (float, optional): Dropout rate for the GNN layers. Defaults to 0.3.
            classifier_dropout (float, optional): Dropout rate for the last layer.
            lm_layer_dropout (float, optional): Dropout rate for the optional `lm_layer`.
            use_roberta (bool): If True, will use RoBERTa for the language model (the one that trains, not the
                one for the embeddings.)

        Raises:
            ValueError: If `n_gnn_layers` is less than 2.
        """
        if n_gnn_layers < 2:
            raise ValueError(f"Argument `n_gnn_layers` must be atleast 2. Was {n_gnn_layers}. ")
        super(QAGNN, self).__init__()

        self.name = model_name
        self.bert = get_bert_model("bert_" + model_name, include_classifier=False, freeze_base_model=freeze_base_model,
                                   freeze_up_to_pooler=freeze_up_to_pooler, use_roberta=use_roberta)

        self.n_gnn_layers = n_gnn_layers
        self.gnn_layers = nn.ModuleList()
        first_gnn_layer = GATConv(self.bert.config.hidden_size, gnn_hidden_dim, dropout=gnn_dropout)
        self.gnn_layers.append(first_gnn_layer)
        for i in range(n_gnn_layers - 2):
            gnn_layer = GATConv(gnn_hidden_dim, gnn_hidden_dim, dropout=gnn_dropout)
            self.gnn_layers.append(gnn_layer)
        last_gnn_layer = GATConv(gnn_hidden_dim, gnn_out_features, dropout=gnn_dropout)
        self.gnn_layers.append(last_gnn_layer)

        claim_dim = self.bert.config.hidden_size
        self.with_lm_layer = False
        if lm_layer_features is not None:
            self.lm_dropout = nn.Dropout(lm_layer_dropout)
            self.lm_layer = nn.Linear(self.bert.config.hidden_size, lm_layer_features)
            claim_dim = lm_layer_features
            self.with_lm_layer = True

        self.gnn_batch_norm = gnn_batch_norm
        if gnn_batch_norm:
            self.gnn_batch_norm_layers = nn.ModuleList()
            for i in range(n_gnn_layers - 1):
                batch_norm_layer = nn.BatchNorm1d(gnn_hidden_dim)
                self.gnn_batch_norm_layers.append(batch_norm_layer)

        self.classsifier_dropout_layer = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(gnn_out_features + claim_dim, 1)

    def forward(self, claim_tokens, data_graphs):
        claim_outputs = self.bert(**claim_tokens)
        claim_embeddings = claim_outputs.last_hidden_state[:, 0]  # Using the [CLS] token's embedding

        batch = data_graphs
        claim_embeddings_expanded = claim_embeddings[batch.batch]  # Expand to match batch size
        relevance_scores = F.cosine_similarity(claim_embeddings_expanded, batch.x, dim=-1).unsqueeze(-1)
        weighted_node_features = batch.x * relevance_scores

        x = weighted_node_features
        for i in range(self.n_gnn_layers):
            x = self.gnn_layers[i](x, batch.edge_index)
            if self.gnn_batch_norm and i < (self.n_gnn_layers - 1):
                x = self.gnn_batch_norm_layers[i](x)
            x = F.relu(x)

        # Pooling the node features
        pooled_gnn_output = global_mean_pool(x, batch.batch)  # Pool over all nodes in each graph

        if self.with_lm_layer:
            claim_embeddings = self.lm_dropout(claim_embeddings)
            claim_embeddings = self.lm_layer(claim_embeddings)

        combined_features = torch.cat((pooled_gnn_output, claim_embeddings), dim=1)

        combined_features = self.classsifier_dropout_layer(combined_features)
        out = self.classifier(combined_features)  # [batch_size, 1]

        return out.squeeze(1)
