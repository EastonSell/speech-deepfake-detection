import pytorch_lightning as pl

from pm.model.model import AudioClassifier


def train_model(train_loader, val_loader=None, test_loader=None, num_classes=2, max_epochs=1):
    model = AudioClassifier(num_classes=num_classes)
    val_loader = val_loader or train_loader
    test_loader = test_loader or val_loader

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    return model
