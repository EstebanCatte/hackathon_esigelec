import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
import matplotlib.pyplot as plt

meta_file = r'batches.meta'
meta_data = unpickle(meta_file)

file = r'data_batch_1'
data_batch_1 = unpickle(file)

print("Label Names:", meta_data['label_names'] )
print(type(meta_data))
print(meta_data.keys()) 

image = data_batch_1['data'][0]
image = image.reshape(3,32,32)
print(image.shape)

image = image.transpose(1,2,0)
print(image.shape)

X_train = data_batch_1['data']
X_train = X_train.reshape(len(X_train),3,32,32)
X_train = X_train.transpose(0,2,3,1)

label_name = meta_data['label_names']
#first image
image = data_batch_1['data'][0]
#first image label index
label = data_batch_1['labels'][0]
#Reshape
image = image.reshape(3,32,32)
#Transpose
image = image.transpose(1,2,0)
#Display the image
plt.imshow(image)
plt.title(label_name[label])
plt.show()
