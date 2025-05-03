import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from networks_lightning import LightningModel
from fmnist import FMNIST
from networks import CNN, MLP
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score


def get_devices():
    """
    Check available device and return preferred one cuda > mps > cpu.
    """
    cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    if len(cuda_devices) > 0:
        return cuda_devices[0]
    else:
        return "cpu"
        
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='whether the model should be trained')
    parser.add_argument('--evaluate', action='store_true', help='whether the model should be evaluated (requires checkpoint)')
    parser.add_argument('--model-dir', type=str, default='model_checkpoints', help='directory for storing/loading models')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint file for evaluation')
    parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs (default=20)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default=0.001)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size (default=32)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default=42)')
    parser.add_argument('--rotate-test', action='store_true', help='Enable rotation augmentation for test data')
    parser.add_argument('--rotate-train', action='store_true', help='Enable rotation augmentation for training data')
    parser.add_argument('--model-type', type=str, choices=['CNN', 'MLP'], default='CNN', help='Type of model to use')
    parser.add_argument('--use-augmentations', action='store_true', help='Enable additional training augmentations')
    parser.add_argument('--analyze-errors', action='store_true', help='Run detailed error analysis on test data')

    
    args = parser.parse_args()

    # Set seed for reproducibility 
    set_seed(args.seed)

    # Load datasets 
    full_train_dataset = FMNIST(train=True, rotate_train=args.rotate_train, use_augmentations=args.use_augmentations)
    test_dataset = FMNIST(train=False, rotate_test=args.rotate_test)

    # Split full_train_dataset into training and validation datasets (80% train, 20% val)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    # Create DataLoaders 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    

    # Initialize model based on selected type
    if args.model_type == 'CNN':
        model_instance = CNN()
    elif args.model_type == 'MLP':
        model_instance = MLP() 
    
    
    early_stopping_callback = EarlyStopping(monitor="val_acc", patience=5)
    

    # Construct the name of logger for tensorboard
    if args.rotate_test and not args.rotate_train:
        name = f"{args.model_type} with test rotation"
    elif args.rotate_test and args.rotate_train:
        name = f"{args.model_type} with train and test rotation"

    elif args.rotate_test and args.use_augmentations:
        name = f"{args.model_type} with test rotation and augmentation"
    else:
        name = f"{args.model_type} without any rotation"

    

    logger = TensorBoardLogger("lightning_logs", name=name)
    
    trainer = Trainer(max_epochs=args.epochs,
                                callbacks=[early_stopping_callback],
                                logger=logger,
                                enable_progress_bar=False) # To make a cleaner output, I turned off the progress bar


    # Train the model if --train flag is provided 
    if args.train:
            lightning_model = LightningModel(model=model_instance)

            trainer.fit(lightning_model, train_loader,val_loader) 

            best_model_path = trainer.checkpoint_callback.best_model_path


    # Evaluate the model if --evaluate flag is provided 
    if args.evaluate:
        if args.checkpoint:
            lightning_model = LightningModel.load_from_checkpoint(args.checkpoint, args.model_type)  # Load specific checkpoint
        else:
            lightning_model = LightningModel.load_from_checkpoint(best_model_path, args.model_type)  # Load best checkpoint from training  
            
        trainer.test(lightning_model, test_loader)

        # If analyzing errors is requested
        if args.analyze_errors:
            all_preds = []
            all_labels = []

            for batch in test_loader:
                images, labels = batch
                outputs = lightning_model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # Calculate balanced accuracy
            bal_acc = balanced_accuracy_score(all_labels, all_preds)
            print(f"Balanced Accuracy: {bal_acc:.4f}")

            # Calculate accuracy per class
            classes = np.unique(all_labels)
            accuracies_per_class = {}
            
            for cls in classes:
                cls_mask = (all_labels == cls)
                acc_cls = np.sum(all_preds[cls_mask] == cls) / np.sum(cls_mask)
                accuracies_per_class[cls] = acc_cls
            
            print("Accuracy per Class:")
            for cls, acc in accuracies_per_class.items():
                print(f"Class {cls}: {acc:.4f}")

            # Show the plot
            plt.figure(figsize=(10, 6))  
            plt.bar(range(len(classes)), list(accuracies_per_class.values()), tick_label=list(classes))
            plt.xlabel('Classes')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Per Class')
            plt.grid(axis='y')

            # Save the plot as an image file in the figures directory
            plt.savefig('figures/accuracy_per_class.png', bbox_inches='tight')