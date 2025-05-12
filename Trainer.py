from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch
from pathlib import Path
import logging

class Trainer:
    """Encapsulates the model training and validation logic."""

    def __init__(self, model, optimizer, loss_fn, scheduler, config, device, writer, logger):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = writer
        self.current_epoch = 0
        
        self.logger = logger


    def _run_epoch(self, dataloader, is_training=True):
        """Runs a single epoch of training or validation."""
        self.model.train(is_training)
        total_loss = 0.0
        phase = "training" if is_training else "validating"
        progress_bar = tqdm(dataloader, desc=f"{phase} Epoch {self.current_epoch}", leave=False)


        correct = 0
        total   = 0

        for x, y, _ in progress_bar:
            x = x.to(self.device)
            y = y.to(self.device).float()        

            if is_training:
                self.optimizer.zero_grad()
            
           
            with torch.set_grad_enabled(is_training):
                outputs = self.model(x)
                outputs = outputs.squeeze(1)

                probs   = torch.sigmoid(outputs)   # (B,) in [0,1]
                preds   = (probs > 0.5).long()    # (B,) in {0,1}

                correct += (preds == y).sum().item()
                total   += y.size(0)
                loss = self.loss_fn(outputs, y)

                if is_training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

       
       
       
       
       
    def train(self, train_dataloader, val_dataloader=None, epochs=1):
        """
        Train the model. Fill in the details of the data loader, the loss function,
        the optimizer, and the training loop.

        Args:
        - train_data_input: Tensor[N_train_samples, C, H, W]
        - train_data_label: Tensor[N_train_samples,]

        Returns:
        - model: torch.nn.Module
        """
        self.logger.info("Starting training...")
        best_val_loss = float("inf")


        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss, _ = self._run_epoch(train_dataloader, is_training=True)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], epoch)

            val_loss = 0.

            if val_dataloader:
                val_loss, val_acc = self._run_epoch(val_dataloader, is_training=False)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("Accuracy/train", _, epoch)
                self.logger.info(f"Epoch {epoch}: train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

                if self.scheduler: # step scheduler, based on validation loss
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()

                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("best_model.pth")
                    self.logger.info(f"Epoch {epoch}: New best validation loss: {best_val_loss:.4f}. Model saved.")
                

            else:
                self.logger.info(f"Epoch {epoch}: train_loss: {train_loss:.4f}")
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(train_loss)
                    elif isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                        self.scheduler.step()

        self.writer.flush()
        self.logger.info("Training complete.")
        self.save_checkpoint("final_model.pth")
        self.logger.info(f"Final model saved.")

    def save_checkpoint(self, filename):

        output_dir = Path(self.config["output_base_dir"])/ "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / filename
        torch.save(self.model.state_dict(),model_path)
