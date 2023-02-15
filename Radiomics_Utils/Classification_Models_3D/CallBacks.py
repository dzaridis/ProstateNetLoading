import tensorflow as tf, numpy as np, os

class SaveModelH5(tf.keras.callbacks.Callback):
    
    def __init__(self, mod_nam, s_path):
        super(SaveModelH5).__init__()
        self.mod_nam = mod_nam
        self.s_path  = s_path
        
    def on_train_begin(self, logs=None):
         self.val_loss = []
         
    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        self.val_loss.append(logs.get("val_loss"))
        if current_val_loss <= min(self.val_loss):
            self.model.save_weights(os.path.join(self.s_path,self.mod_nam) , save_format='tf')

class EarlyStoppingMinLoss(tf.keras.callbacks.Callback):
    """Stop training at minimum loss with a patience of desired epocks
    Args:
        patience : Number of epochs to wait when the minimum loss has been achieved
    """
    def __init__(self,patience = 0):
        def __init__(self, patience=0):
            super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

