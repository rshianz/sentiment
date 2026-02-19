import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

def plots(df):
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='polarity', data=df, order=['positive', 'negative', 'neutral'])
    plt.title('Sentiment Distribution')
    plt.xlabel('polarity')
    plt.ylabel('number')
    plt.show()

    df['word_count'] = df['Sentence'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=30, kde=True, color='purple')
    plt.title('Word Count Histogram')
    plt.xlabel('word_count')
    plt.ylabel('quantity')
    plt.show()

    max_len = df['word_count'].max()
    print(max_len)

    all_aspects = " ".join(df['Aspect Term'].astype(str))
    wordcloud = WordCloud(
        background_color='white',
        width=1600,
        height=800,
        max_words=100
    ).generate(all_aspects)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Aspects')
    plt.show()
class RestaurantDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = RestaurantDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, model_name):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.out(self.drop(outputs.pooler_output))


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.float() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.float() / n_examples, np.mean(losses)

if __name__ == '__main__':
    MAX_LEN = 80
    BATCH_SIZE = 16
    MODEL_NAME = 'bert-base-uncased'
    EPOCHS = 4
    train_file_path = 'dataset/Restaurants_Train_v2.csv' #i rename "archive 2" to dataset
    test_file_path = 'dataset/Restaurants_Test_Data_PhaseB.csv'

    df = pd.read_csv(train_file_path)
    df.columns = df.columns.str.strip()
    plots(df)

    df['text'] = df['Sentence'] + " [SEP] " + df['Aspect Term']
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['polarity'].map(label_map)
    df = df.dropna(subset=['label'])

    df['label'] = df['label'].astype(int)

    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df.label
    )
    print(f"train set size: {len(df_train)}")
    print(f"val set size: {len(df_val)}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    print("dataloader created.")

    if torch.cuda.is_available(): # no need to change for linux and windows or colab
        device = torch.device("cuda:0")
        print("Found CUDA GPU, using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Found Apple Silicon GPU, using MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU.")
    print(f"Using device: {device}")

    model = SentimentClassifier(len(label_map), MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    print("\nstart training...\n" + "*" * 23)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'train loss {train_loss:.4f} | accuracy {train_acc:.4f}')

        val_acc, val_loss = eval_model(
            model,
            val_loader,
            loss_fn,
            device,
            len(df_val)
        )

        print(f'val loss   {val_loss:.4f} | accuracy {val_acc:.4f}')
        print()

        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
            print(f"=> best model saved with acc : {best_accuracy:.4f}!")

    print("train finished.")
