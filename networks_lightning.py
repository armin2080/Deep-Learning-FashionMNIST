import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from networks import CNN, MLP

class LightningModel(pl.LightningModule):
    def __init__(self, model, lr=0.001):
        super(LightningModel, self).__init__()
        self.model = model
        self.lr = lr
        
        # Initialize accuracy metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=6)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=6)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=6)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        
        loss = F.cross_entropy(outputs, labels)
        
        acc = self.train_accuracy(outputs.softmax(dim=-1), labels)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        loss = F.cross_entropy(outputs, labels)
        
        acc = self.val_accuracy(outputs.softmax(dim=-1), labels)

        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        loss = F.cross_entropy(outputs, labels)
    
        acc = self.test_accuracy(outputs.softmax(dim=-1), labels)

        self.log('test_loss', loss)
        self.log('test_acc', acc)

    # Load the checkpoint
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, model_type):
        
        checkpoint = torch.load(checkpoint_path)

        # Reinitialize model instance
        if model_type == "CNN":
            model_instance = CNN()

        elif model_type == "MLP":
            model_instance = MLP()
        
        lightning_model = cls(model=model_instance)

        # Load state dict
        lightning_model.load_state_dict(checkpoint['state_dict'], strict=False)  
        
        return lightning_model