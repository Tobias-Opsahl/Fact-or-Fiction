import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import GATConv
from transformers import AutoModelForSequenceClassification, BertModel, BertTokenizer


def get_bert_model(model_name, num_labels=2, freeze=False, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", cache_dir="./cache", trust_remote_code=True, num_labels=num_labels,
        output_hidden_states=True
    ).to(device)

    model.name = model_name
    if freeze:
        for params in model.base_model.parameters():
            params.requires_grad = False
    return model


def calculate_all_relevances(question_embedding, graph_data):
    relevance_scores = cosine_similarity(question_embedding.unsqueeze(1), graph_data.x.unsqueeze(0), dim=2)
    return relevance_scores


class QAGNN(nn.Module):
    def __init__(self, model_name, hidden_dim, out_features, bert_model_name="bert-base-uncased",
                 dropout=0.6):
        super(QAGNN, self).__init__()
        self.name = model_name
        self.bert = BertModel.from_pretrained(bert_model_name)

        self.gnn = GATConv(self.bert.config.hidden_size, hidden_dim, dropout=dropout)
        self.gnn_out = GATConv(hidden_dim, out_features, dropout=dropout)

        self.classifier = nn.Linear(out_features + self.bert.config.hidden_size, 1)

    def forward(self, claim_tokens, data_graph):

        claim_outputs = self.bert(**claim_tokens)
        claim_embedding = claim_outputs.last_hidden_state[:, 0]  # Using the [CLS] token's embedding

        relevance_scores = calculate_all_relevances(claim_embedding, data_graph)

        weighted_node_features = data_graph.x * relevance_scores.unsqueeze(2)

        # TODO: Vectorize
        batch_output = []
        for i in range(weighted_node_features.size(0)):
            x = self.gnn(weighted_node_features[i], data_graph.edge_index)
            x = F.relu(x)
            x = self.gnn_out(x, data_graph.edge_index)
            x = torch.mean(x, dim=0, keepdim=True)  # [1, feature_size]
            batch_output.append(x)

        batch_output = torch.cat(batch_output, dim=0)  # [batch_size, feature_size]
        combined_features = torch.cat((batch_output, claim_embedding), dim=1)  # [batch_size, feature_size+hidden_size]
        out = self.classifier(combined_features)  # [batch_size, 1]

        return out.squeeze()


if __name__ == "__main__":
    # Load Pre-trained BERT Model and Tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
