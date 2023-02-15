import tensorflow as tf

class ConvLayer3D(tf.keras.layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides, padding='same',isbn = True, isdp = True):
        super(ConvLayer3D, self).__init__()
        # conv ConvLayer2D
        self.conv = tf.keras.layers.Conv3D(
                    kernel_num, 
                    kernel_size=kernel_size, 
                    strides=strides, 
                    padding=padding
                )
        # batch norm layer
        self.bn   = tf.keras.layers.BatchNormalization()
        # Dropout layer
        self.dp    = tf.keras.layers.Dropout(rate = 0.3)
        self.isbn = isbn
        self.isdp = isdp
        
    def call(self, input_tensor, training=True):
        x = self.conv(input_tensor)
        if self.isbn == True:
            x = self.bn(x, training=training)
        if self.isdp == True:
            x = self.dp(x) 
        x = tf.nn.relu(x)
        return x
    
class MaxPooling(tf.keras.layers.Layer):
    def __init__(self, max=True):
        super(MaxPooling, self).__init__()

        # pooling layer 
        
        self.maxpool  = tf.keras.layers.MaxPooling3D(
                  pool_size=(3, 3, 3), 
                  strides=(2, 2, 2),
                  padding = "same")
        self.avepool  = tf.keras.layers.AveragePooling3D(
                  pool_size=(3, 3, 3), 
                  strides=(2, 2, 2),
                  padding = "same")
        self.max = True


    def call(self, input_tensor, training=True):
        # forward pass 
        if self.max == True:
            pool_x = self.maxpool(input_tensor)
        else:
            pool_x = self.avepool(input_tensor)

        return pool_x
    
class ConvModule(tf.keras.layers.Layer):
    def __init__(self,kernel_num, kernel_size, strides, padding='same',isbn = True, isdp = True, max = True):
        super(ConvModule,self).__init__()
        
        self.conv1  = ConvLayer3D(kernel_num, kernel_size, strides, padding='same',isbn=isbn, isdp=isdp)
        self.conv2  = ConvLayer3D(kernel_num, kernel_size, strides, padding='same',isbn=isbn, isdp=isdp)
        self.pool   = MaxPooling(max = max)
        
    def call(self, input_tensor, training=True):
            # forward pass 
        x1 = self.conv1(input_tensor)
        x2 = self.conv2(x1)
        x3 = self.pool(x2)

        return x3
    
class ResConvModule(tf.keras.layers.Layer):
    def __init__(self,kernel_num, kernel_size, strides, padding='same',isbn = True, isdp = True, max = True):
        super(ResConvModule,self).__init__()
        
        self.conv1  = ConvLayer3D(kernel_num, 
                                  kernel_size, 
                                  strides, 
                                  padding='same',
                                  isbn=isbn, 
                                  isdp=isdp)
        
        self.conv2  = ConvLayer3D(kernel_num,
                                  kernel_size, 
                                  strides, 
                                  padding='same',
                                  isbn=isbn, 
                                  isdp=isdp)
        
        self.pool   = MaxPooling(max = max)
        self.add    = tf.keras.layers.Add()
        self.ident  = tf.keras.layers.Conv3D(
                    kernel_num, 
                    kernel_size=1, 
                    strides=1, 
                    padding=padding)
        
    def call(self, input_tensor, training=True):
            # forward pass 
        
        x1 = self.conv1(input_tensor)
        x2 = self.conv2(x1)
        x_identity = self.ident(input_tensor)
        x3 = self.add([x_identity,x2])
        x3 = self.pool(x2)

        return x3

class Flatten_Layer(tf.keras.layers.Layer):
    def __init__(self,Flat_Method = "GAP"):
        """
        self
        Flat_Method: "GAP" or "Flat" or "GMP" 
        """
        super(Flatten_Layer,self).__init__()
        self.gap         = tf.keras.layers.GlobalAveragePooling3D()
        self.flat        = tf.keras.layers.Flatten()
        self.gmp         = tf.keras.layers.GlobalMaxPooling3D()
        self.Flat_Method = Flat_Method
        
        
    def call(self, input_tensor, training=True):
            # forward pass 
        x = input_tensor
        
        if self.Flat_Method =="GAP":
            x = self.gap(x)
        elif self.Flat_Method =="GMP":
            x = self.gmp(x)
        else:
            x = self.flat(x)
        
            
        return x
    
class DenseModule(tf.keras.layers.Layer):
    def __init__(self,Dense_NN = [256,128,64]):
        """
        self
        Dense_NN   : list of Dense Neurons, Default [256,128,64] 
        """
        super(DenseModule,self).__init__()
        self.Dense_NN    = Dense_NN
        
    def call(self, input_tensor, training=True):
            # forward pass 
        x = input_tensor
        
        for neurons in self.Dense_NN:
            x = tf.keras.layers.Dense(neurons,activation="relu")(x)
            
        return x