# Install dependencies
!pip install datasets evaluate rouge_score tensorboard

# Import dependencies
from transformers import AutoTokenizer, LEDConfig, LEDForConditionalGeneration, get_linear_schedule_with_warmup
from datasets import load_dataset
import evaluate
import torch
from torch.utils.data import Dataset, DataLoader
import time
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
multi_news = load_dataset("multi_news")

# Instantiate model
tokenizer = AutoTokenizer.from_pretrained('allenai/PRIMERA')
config = LEDConfig.from_pretrained('allenai/PRIMERA')
model = LEDForConditionalGeneration.from_pretrained('allenai/PRIMERA', config=config)
model.gradient_checkpointing_enable()
model.to(device)

# Load metric
rouge = evaluate.load('rouge')

# Preprocess data
torch.manual_seed(1)
torch.cuda.manual_seed(1)
class MultiNews(Dataset):
  def __init__(self, dataset, split):
    self.split = split
    self.dataset = dataset[self.split]
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self, idx):
    document = multi_news[self.split][idx]["document"]
    document = document.replace("\n ", "")
    docs = document.split(" ||||| ")
    document = [tokenizer.decode(tokenizer.encode(doc, truncation=True, max_length=1024 // len(docs))[1:-1]) for doc in docs]
    document = "<doc-sep>".join(document)
    summary = multi_news[self.split][idx]["summary"]
    summary = summary[2:]
    return {"document": document, "summary": summary}

train_dataset = torch.utils.data.Subset(MultiNews(multi_news, "train"), range(0, 7))
train_dataloader = DataLoader(train_dataset, 2, True)

validation_dataset = torch.utils.data.Subset(MultiNews(multi_news, "validation"), range(0, 7))
validation_dataloader = DataLoader(train_dataset, 2, True)

test_dataset = MultiNews(multi_news, "test")
test_dataloader = DataLoader(train_dataset, 2, True)

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# Set up scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2500, num_training_steps=25000)

# Training loop
torch.manual_seed(1)
torch.cuda.manual_seed(1)
writer = SummaryWriter()
# start = time.perf_counter()
for epoch in range(4):
  model.train()
  train_loss = []
  # print(f"Epoch: {epoch+1}\n")
  for step, data in enumerate(train_dataloader):
    X = data["document"]
    y = data["summary"]
    input_ids = torch.stack([tokenizer(x, return_tensors="pt", padding="max_length", max_length=1024)["input_ids"].squeeze().int() for x in X], dim=0).to(device)
    labels = torch.stack([tokenizer(x, return_tensors="pt", padding="max_length", max_length=1024)["input_ids"].squeeze().int() for x in y], dim=0).type(torch.LongTensor).to(device)
    attention_mask = torch.stack([tokenizer(x, return_tensors="pt", padding="max_length", max_length=1024)["attention_mask"].squeeze().int() for x in X], dim=0).to(device)
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1
    global_attention_mask[input_ids == tokenizer.encode("<doc-sep>")[1:-1][0]] = 1
    global_attention_mask = global_attention_mask.to(device)
    preds = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    loss = preds.loss
    train_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    # print(f"Batch: {step+1}/{len(train_dataloader)} | Train Loss: {loss.item()}")
  writer.add_scalar("Loss/train", loss.item(), epoch+1)
  model.eval()
  with torch.inference_mode():
    validation_loss = []
    validation_rouge_1 = []
    validation_rouge_2 = []
    validation_rouge_l = []
    for step, data in enumerate(validation_dataloader):
      X = data["document"]
      y = data["summary"]
      input_ids = torch.stack([tokenizer(x, return_tensors="pt", padding="max_length", max_length=1024)["input_ids"].squeeze().int() for x in X], dim=0).to(device)
      labels = torch.stack([tokenizer(x, return_tensors="pt", padding="max_length", max_length=1024)["input_ids"].squeeze().int() for x in y], dim=0).type(torch.LongTensor).to(device)
      attention_mask = torch.stack([tokenizer(x, return_tensors="pt", padding="max_length", max_length=1024)["attention_mask"].squeeze().int() for x in X], dim=0).to(device)
      global_attention_mask = torch.zeros_like(input_ids)
      global_attention_mask[:, 0] = 1
      global_attention_mask[input_ids == tokenizer.encode("<doc-sep>")[1:-1][0]] = 1
      global_attention_mask = global_attention_mask.to(device)
      preds = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
      loss = preds.loss
      validation_loss.append(loss.item())
      generated_ids = model.generate(input_ids=input_ids, global_attention_mask=global_attention_mask, attention_mask=attention_mask, use_cache=True, max_length=1024, num_beams=5)
      generated_str = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
      score=rouge.compute(predictions=generated_str, references=y)
      validation_rouge_1.append(score["rouge1"])
      validation_rouge_2.append(score["rouge2"])
      validation_rouge_l.append(score["rougeL"])
    # print(f"\nValidation Loss: {np.mean(validation_loss)} | ROUGE-1: {np.mean(validation_rouge_1)} | ROUGE-2: {np.mean(validation_rouge_2)} | ROUGE-L: {np.mean(validation_rouge_l)}\n")
    writer.add_scalar("Loss/validation", np.mean(validation_loss), epoch+1)
    writer.add_scalar("Rouge/1", np.mean(validation_rouge_1), epoch+1)
    writer.add_scalar("Rouge/2", np.mean(validation_rouge_2), epoch+1)
    writer.add_scalar("Rouge/L", np.mean(validation_rouge_l), epoch+1)
# end = time.perf_counter()
# print(datetime.timedelta(seconds=(end-start)*((len(train_dataloader)+len(validation_dataloader))*9+len(test_dataloader))))
writer.flush()
writer.close()

%load_ext tensorboard
%tensorboard --logdir=runs

# generate samples with diverse beam searc
samples = [tokenizer.batch_decode(model.generate(input_ids[i].unsqueeze(dim=0), max_length=256, num_beams=16, num_return_sequences=16, num_beam_groups=4, diversity_penalty=0.5).tolist(), skip_special_tokens=True) for i in range(len(input_ids))]
samples

# Get rouge score
generated_ids = model.generate(input_ids=input_ids, global_attention_mask=global_attention_mask, attention_mask=attention_mask, use_cache=True, max_length=1024, num_beams=5)
generated_str = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
score=rouge.compute(predictions=generated_str, references=y)
score["rouge1"], score["rouge2"], score["rougeL"], score["rougeLsum"]