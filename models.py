import torch
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


if __name__ == "__main__":
    # Load Pre-trained BERT Model and Tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
