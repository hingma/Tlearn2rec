import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

class TrainingVisualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'epochs': [],
            'learning_rates': []
        }
        
    def update(self, epoch, train_loss, valid_loss, lr):
        """Update training history with new metrics"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['valid_loss'].append(valid_loss)
        self.history['learning_rates'].append(lr)
        
    def plot_training_curves(self, title=None):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epochs'], self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['epochs'], self.history['valid_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if title:
            plt.title(title)
        else:
            plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'training_curves.png')
        plt.close()
        
    def plot_learning_rate(self):
        """Plot learning rate changes over time"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epochs'], self.history['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(self.save_dir / 'learning_rate.png')
        plt.close()
        
    def save_history(self):
        """Save training history to JSON file"""
        history_dict = {
            'epochs': self.history['epochs'],
            'train_loss': [float(x) for x in self.history['train_loss']],  # Convert numpy floats to Python floats
            'valid_loss': [float(x) for x in self.history['valid_loss']],
            'learning_rates': [float(x) for x in self.history['learning_rates']]
        }
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=4)
            
    def load_history(self, filepath):
        """Load training history from JSON file"""
        with open(filepath, 'r') as f:
            history_dict = json.load(f)
        self.history = history_dict
