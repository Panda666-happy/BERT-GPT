from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 初始化设置
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据准备（示例数据，实际应替换为真实数据集）
data = {
    "text": [
        "是近年来最值得一看的国产佳作，强烈推荐！",
        "分量太少了，照片看着满满的，实际就几口。",
        "剧情精彩，演员演技在线",
        "配送慢，包装还破损了"
    ],
    "label": [1, 0, 1, 0]  # 1:正面, 0:负面
}
df = pd.DataFrame(data)

# 3. 数据集划分
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


# 4. 自定义Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 5. 创建DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = SentimentDataset(
        texts=df.text.values,
        labels=df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size)


BATCH_SIZE = 16
MAX_LEN = 128
train_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)

# 6. 模型初始化
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
).to(device)

# 7. 训练配置
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# 8. 训练函数
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(data_loader)


# 9. 验证函数
def eval_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset)


# 10. 训练循环
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
    val_acc = eval_model(model, val_loader, device)

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train loss: {train_loss:.4f}')
    print(f'Val accuracy: {val_acc:.4f}')
    print('-' * 50)


# 11. 预测函数（微调后使用）
def predict_sentiment(text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    return "正面" if predicted_label == 1 else "负面"


# 测试预测
print(f"影评情感倾向：{predict_sentiment('是近年来最值得一看的国产佳作，强烈推荐！')}")
print(f"外卖评价情感倾向：{predict_sentiment('分量太少了，照片看着满满的，实际就几口。')}")