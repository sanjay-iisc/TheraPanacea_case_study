import tensorflow as tf
import os
class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, checkpoint_path, save_freq=1):
        super(CheckpointCallback, self).__init__()
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.save_freq = 2
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_file = os.path.join(self.checkpoint_path, f"epoch_{epoch+1}.ckpt")
            self.model.save_weights(checkpoint_file)
            print(f"Checkpoint saved at epoch {epoch+1}")