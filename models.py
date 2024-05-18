import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoModelForSequenceClassification, BertModel
from utils import get_logger

logger = get_logger(__name__)


def get_bert_model(model_name="bert", include_classifier=True, num_labels=2, freeze_base_model=False,
                   freeze_up_to_pooler=True, dropout_rate=0):
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

    Returns:
        transformer model: The loaded model.
    """
    if freeze_base_model and freeze_up_to_pooler:
        logger.warn("Both `freeze_base_model` and `freeze_up_to_pooler` is True. Freezing base model.")

    if include_classifier:
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", cache_dir="./cache", trust_remote_code=True, num_labels=num_labels,
            output_hidden_states=True
        )
    else:
        model = BertModel.from_pretrained("bert-base-uncased")

    model.name = model_name
    if freeze_base_model:
        for params in model.base_model.parameters():
            params.requires_grad = False
    elif freeze_up_to_pooler:
        for name, params in model.base_model.parameters():
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
    def __init__(self, model_name, n_gnn_layers=2, gnn_hidden_dim=256, gnn_out_features=256, gnn_batch_norm=True,
                 freeze_base_model=False, freeze_up_to_pooler=True, gnn_dropout=0.3, classifier_dropout=0.2,
                 vectorized=True):
        """
        Args:
            model_name (str): Name of the model, will be saved as a class variable.
            n_gnn_layers (int, optional): Number of layers in the GNN. Defaults to 2.
            gnn_hidden_dim (int, optional): Number of nodes in GNN layers. Defaults to 256.
            gnn_out_features (int, optional): Number of output nodes from the GNN. Defaults to 256.
            gnn_batch_norm (bool, optional): Whether or not to apply batch norm between the GNN layers.
                Defaults to True.
            freeze_base_model (bool, optional): Freeze the base model of the Bert language model. Defaults to False.
            freeze_up_to_pooler (bool, optional): Freeze up to the last part of the Bert model. Defaults to True.
            gnn_dropout (float, optional): Dropout rate for the GNN layers. Defaults to 0.3.
            classifier_dropout (float, optiona): Dropout rate for the last layer.

        Raises:
            ValueError: If `n_gnn_layers` is less than 2.
        """
        if n_gnn_layers < 2:
            raise ValueError(f"Argument `n_gnn_layers` must be atleast 2. Was {n_gnn_layers}. ")
        super(QAGNN, self).__init__()
        self.vectorized = vectorized

        self.name = model_name
        self.bert = get_bert_model("bert_" + model_name, include_classifier=False, freeze_base_model=freeze_base_model,
                                   freeze_up_to_pooler=freeze_up_to_pooler)

        self.n_gnn_layers = n_gnn_layers
        self.gnn_layers = nn.ModuleList()
        first_gnn_layer = GATConv(self.bert.config.hidden_size, gnn_hidden_dim, dropout=gnn_dropout)
        self.gnn_layers.append(first_gnn_layer)
        for i in range(n_gnn_layers - 2):
            gnn_layer = GATConv(gnn_hidden_dim, gnn_hidden_dim, dropout=gnn_dropout)
            self.gnn_layers.append(gnn_layer)
        last_gnn_layer = GATConv(gnn_hidden_dim, gnn_out_features, dropout=gnn_dropout)
        self.gnn_layers.append(last_gnn_layer)

        self.gnn_batch_norm = gnn_batch_norm
        if gnn_batch_norm:
            self.gnn_batch_norm_layers = nn.ModuleList()
            for i in range(n_gnn_layers - 1):
                batch_norm_layer = nn.BatchNorm1d(gnn_hidden_dim)
                self.gnn_batch_norm_layers.append(batch_norm_layer)

        self.classsifier_dropout_layer = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(gnn_out_features + self.bert.config.hidden_size, 1)

    def forward(self, claim_tokens, data_graphs):

        claim_outputs = self.bert(**claim_tokens)
        claim_embeddings = claim_outputs.last_hidden_state[:, 0]  # Using the [CLS] token's embedding

        if not self.vectorized:
            # Unvectorized GNN processing.
            batch_output = []
            batch_size = claim_embeddings.shape[0]
            for i in range(batch_size):
                claim_embedding = claim_embeddings[i]
                data_graph = data_graphs[i]
                # Calculate relevance score
                relevance_scores = cosine_similarity(claim_embedding.unsqueeze(0), data_graph.x, dim=1)
                weighted_node_features = data_graph.x * relevance_scores.unsqueeze(1)

                # Iterate through GNN layers
                x = weighted_node_features
                for j in range(self.n_gnn_layers):
                    x = self.gnn_layers[j](x, data_graph.edge_index)
                    if self.gnn_batch_norm and j < (self.n_gnn_layers - 1):
                        x = self.gnn_batch_norm_layers[j](x)
                    x = F.relu(x)
                pooled_gnn_output = torch.mean(x, dim=0, keepdim=True)  # Pool all of the nodes
                batch_output.append(pooled_gnn_output)
            batch_output = torch.cat(batch_output, dim=0)  # [batch_size, gnn_out_features]
            combined_features = torch.cat((batch_output, claim_embeddings), dim=1)
        else:  # Vectorized
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
            combined_features = torch.cat((pooled_gnn_output, claim_embeddings), dim=1)

        combined_features = self.classsifier_dropout_layer(combined_features)
        out = self.classifier(combined_features)  # [batch_size, 1]
        return out.squeeze()
