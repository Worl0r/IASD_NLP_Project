import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers_project.linformer.linformer import LinformerEnc


# Transformer-based classifier
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=2,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = (
            torch.arange(0, seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        cls_token = x[:, 0, :]  # Use [CLS] token
        logits = self.classifier(cls_token)
        return logits


# Load SST-2 dataset
def load_data(tokenizer, max_length=64):
    dataset = load_dataset("glue", "sst2")

    def tokenize(example):
        return tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return dataset


def train(
    model, dataloader, optimizer, scheduler, loss_fn, writer, device, epoch
):
    model.train()
    total_loss, total_acc = 0, 0

    for i, batch in enumerate(
        tqdm(dataloader, desc=f"Training Epoch {epoch}")
    ):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.squeeze(-1), labels.to(float))
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        writer.add_scalar("Loss/train_t", loss, epoch + i * len(dataloader))
        writer.add_scalar("Accuracy/train_t", acc, epoch + i * len(dataloader))

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", avg_acc, epoch)


def validate(model, dataloader, loss_fn, writer, device, epoch):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = loss_fn(logits.squeeze(-1), labels.to(float))

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", avg_acc, epoch)
    print(f"[Epoch {epoch}] Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.4f}")


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_data(tokenizer)

    # Configuration
    batch_size = 32
    num_epochs = 10

    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = TransformerClassifier(
    #     vocab_size=tokenizer.vocab_size, num_classes=2
    # ).to(device)

    model = LinformerEnc(
        seq_len=64,
        dim=256,
        dim_lin_base=256,
        vocab_size=tokenizer.vocab_size,
        n_features=1,
        device=device,
        d_conversion=128,
        max_prediction_length=1,
        dropout_input=0.01,
        dropout_multi_head_att=0.01,
        dropout_lin_att=0.01,
        dim_ff=256,
        ff_intermediate=None,
        dropout_ff=0.01,
        nhead=4,
        n_layers=4,
        dropout=0.01,
        parameter_sharing="layerwise",
        k_reduce_by_layer=0,
        method="learnable",
        activation="gelu",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter("runs/transformer_classifier")

    for epoch in range(1, num_epochs + 1):
        train(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            writer,
            device,
            epoch,
        )
        validate(model, val_loader, loss_fn, writer, device, epoch)

    writer.close()


if __name__ == "__main__":
    main()
