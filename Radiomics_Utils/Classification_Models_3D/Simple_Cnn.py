import tensorflow as tf
from PCA_Utilities.Classification_Models_3D.Util_Layers import *

class SimpleCNN(tf.keras.Model):
    def __init__(self, num_cls = 1, num_filters = [16,32,64,128,256]):
        super(SimpleCNN,self).__init__()
        
        self.clsfier = tf.keras.layers.Dense(num_cls, activation='sigmoid')
        self.num_filters = num_filters
        self.num_cls = 1
        self.conv = []
        
        for filt in num_filters:
            self.conv.append(ConvModule(kernel_num = filt, kernel_size = 3, 
                           strides = 1, padding='same',isbn = True, 
                           isdp = True, max = True))
            
        self.Flat_Method = tf.keras.layers.Flatten()
        self.Dense       = tf.keras.layers.Dense(64)
        self.Dense_NN = tf.keras.layers.Dense(num_cls,activation='sigmoid')
        
        
    def call(self, input_tensor, training=True, **kwargs):
        # forward pass 
        x = input_tensor
        
        for cnv in self.conv:
            x = cnv(x)
            
        x = self.Flat_Method(x)
        x = self.Dense(x)
        x = self.Dense_NN(x)

        return x
    
    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class Feature_Extractor_CNN(tf.keras.Model):
    def __init__(self, num_cls = 1, num_filters = [16,32,64,128,256],Flat_Method = "GAP"):
        super(Feature_Extractor_CNN,self).__init__()
        
        self.num_filters = num_filters
        self.num_cls = 1
        self.Flat_Method = Flat_Method
    def call(self, input_tensor, training=False, **kwargs):
        # forward pass 
        x = input_tensor
        for filt in self.num_filters:
            x = ConvModule(kernel_num = filt, kernel_size = 3, 
                           strides = 1, padding='same',isbn = True, 
                           isdp = True, max = True)(x)
        x = Flatten_Layer(Flat_Method=self.Flat_Method)(x)

        return x
    
    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))