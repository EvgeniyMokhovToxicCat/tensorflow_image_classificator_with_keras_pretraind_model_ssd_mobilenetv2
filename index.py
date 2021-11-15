import tensorflow as tf 
import os
import numpy as np
import matplotlib.pyplot as plt

##########################################################################################################
# Off Warnings
import warnings
warnings.filterwarnings('ignore')

# Stop annoying tensorflow warning messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Switch working directory to file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

##########################################################################################################
## Ckecking TF Version 
temp = tf.__version__
temp = int(temp.replace('.', ''))
if temp < 260:
    print('Ckeck TF Version ... ERROR:', temp)
    exit()
else:
    print('Ckeck TF Version ... ', 'OK')

##########################################################################################################
# Enable GPU dynamic memory allocation
temp = tf.config.list_physical_devices('GPU')    
if len(temp) == 0:
    print('Ckeck GPU ... FAILED:', tf.config.list_physical_devices('GPU') )
else:
    print('Ckeck GPU,', 'found:', len(temp), '...', 'OK')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print('Check GPU and set Memory Growth ... ERROR:' . e)
        else:
            print('Check GPU and set Memory Growth ... OK:', gpu)

##########################################################################################################
# Loading Pretrained Model 
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

##########################################################################################################
# Load Image 
img_path = 'images/cat.png'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

##########################################################################################################
# Make as prediction
preds = model.predict(x)
prediction_dict = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)
print('Predicted:', prediction_dict)

##########################################################################################################
# Show Result 
plt.text(0, 0, prediction_dict[0][0][1], bbox=dict(fill=True, edgecolor='red', linewidth=2))
plt.imshow(img)
plt.show()
