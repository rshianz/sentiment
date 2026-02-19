import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
class_name = ['negative', 'neutral', 'positive']

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

def predict_sentiment(text, aspect_term, model, tokenizer, device, max_len=128):
    model.eval()
    encoded_text = f"{text} [SEP] {aspect_term}"
    encoding = tokenizer(
        encoded_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        _, prediction_index = torch.max(output, dim=1)
    return class_name[prediction_index.item()]

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # you change mps to cuda if you using linux or windows
    print("device:", device)

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = SentimentClassifier(n_classes=len(class_name))
    model.load_state_dict(torch.load('best_model_state.bin', map_location=device))
    model = model.to(device)
    print("*"*20)
    print("sample: \n")
    text1 = "The food was delicious but the service was slow."
    print(f"\nSentence: '{text1}'")
    print(f"Aspect: 'food' -> Predicted: '{predict_sentiment(text1, 'food', model, tokenizer, device)}'")
    print(f"Aspect: 'service' -> Predicted: '{predict_sentiment(text1, 'service', model, tokenizer, device)}'")

    text2 = "I think the price is a bit too high for what you get."
    print(f"\nSentence: '{text2}'")
    print(f"Aspect: 'price' -> Predicted: '{predict_sentiment(text2, 'price', model, tokenizer, device)}'")
    print("*"*20)

    print("\nInteractive mode: Type 'exit' to quit.")
    while True:
        try:
            user_text = input("Enter a sentence: ")
            if user_text.lower() == 'exit':
                break
            user_aspect = input("Enter the aspect term: ")
            if user_aspect.lower() == 'exit':
                break

            prediction = predict_sentiment(user_text, user_aspect, model, tokenizer, device)
            print(f"--> Predicted Sentiment: {prediction}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

