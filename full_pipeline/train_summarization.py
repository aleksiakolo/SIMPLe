import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import BartTokenizer
from torchmetrics.text.rouge import ROUGEScore
from full_pipeline.models.summary import LegalBERTEncoder, LegalBERTSeq2Seq, get_dataloader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig
import wandb

class LegalBERTSeq2SeqPL(pl.LightningModule):
    def __init__(self, encoder, tokenizer, learning_rate=3e-5, max_output_length=512):
        """
        Lightning Module for LegalBERT Seq2Seq with Cross-Attention Encoder and Decoder.
        """
        super().__init__()
        self.model = LegalBERTSeq2Seq(encoder)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.max_output_length = max_output_length

        # Metrics
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.rouge = ROUGEScore()
        self.training_losses = []
        self.validation_outputs = []

        self.save_hyperparameters()

    def forward(self, batch):
        """
        Forward pass through the model.
        Args:
            batch (dict): Dictionary containing sources, source_attention_mask, input_ids, and attention_mask.
        Returns:
            torch.Tensor: Logits for next-token prediction.
        """
        return self.model(
            sources=batch["sources"],
            source_attention_mask=batch["source_attention_mask"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )


    def training_step(self, batch, batch_idx):
        """
        Training step: Compute loss and log metrics.
        Args:
            batch (dict): Input batch for training.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Loss for the current training step.
        """
        # Forward pass
        logits = self(batch)
        
        # Compute loss
        labels = batch["input_ids"]
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.training_losses.append(loss.item())
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss


    def on_train_epoch_end(self):
        """
        End of training epoch: Log aggregated training loss to WandB.
        """
        avg_train_loss = torch.tensor(self.training_losses).mean().item()
        wandb.log({"train_epoch_loss": avg_train_loss, "epoch": self.current_epoch})
        self.training_losses = []  # Reset training losses

    def validation_step(self, batch, batch_idx):
        """
        Validation step: Compute loss, perplexity, and ROUGE scores.
        Args:
            batch (dict): Input batch for validation.
            batch_idx (int): Index of the batch.
        Returns:
            dict: Validation metrics for the current step.
        """
        # Forward pass
        logits = self(batch)

        # Compute loss
        labels = batch["input_ids"]
        val_loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Compute perplexity
        perplexity = torch.exp(val_loss)

        # Generate predictions
        generated_ids = self.model.generate(
            sources=batch["sources"],
            source_attention_mask=batch["source_attention_mask"],
            max_length=self.max_output_length,
            num_beams=4,
        )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels_decoded = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        rouge_scores = self.rouge(preds, labels_decoded)

        # Log metrics
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log("val_perplexity", perplexity, prog_bar=True, logger=True)
        self.log_dict({f"val_{k}": v for k, v in rouge_scores.items()}, prog_bar=True, logger=True)

        # Store validation outputs for epoch-end processing
        self.validation_outputs.append({
            "val_loss": val_loss.item(),
            "val_perplexity": perplexity.item(),
            **rouge_scores,
        })
        return {"val_loss": val_loss, "val_perplexity": perplexity, **rouge_scores}


    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

@hydra.main(version_base=None, config_path="./configs", config_name="summarization")
def main(cfg: DictConfig):
    # Tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    # Encoder and Decoder
    encoder = LegalBERTEncoder(
        embedding_dim=cfg.model.embedding_dim,
        num_heads=cfg.model.num_heads,
        ff_dim=cfg.model.ff_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    
    # Model
    model = LegalBERTSeq2SeqPL(
        encoder=encoder,
        tokenizer=tokenizer,
        learning_rate=cfg.training.learning_rate,
        max_output_length=cfg.model.max_output_length,
    )

    # DataLoaders
    train_dataloader = get_dataloader(cfg.data.train.data_dir, cfg.data.train.labels_dir, cfg.training.batch_size)
    val_dataloader = get_dataloader(cfg.data.val.data_dir, cfg.data.val.labels_dir, cfg.training.batch_size)

    # WandB Logger
    wandb_logger = WandbLogger(project=cfg.project_name)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.callbacks.checkpoint.dirpath,
        filename="best-checkpoint",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.callbacks.early_stopping.patience,
        verbose=True,
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu",
        devices=cfg.training.gpus,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
