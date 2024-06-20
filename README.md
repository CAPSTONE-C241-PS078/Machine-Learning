<h1 align="center">CatarActNow - Machine Learning Documentation</h1>
<p align="center">
  <img src="https://github.com/CAPSTONE-C241-PS078/Machine-Learning/assets/99121625/68274782-374f-42de-b450-7d672f6be9e7/overview.png" alt="Deskripsi Gambar" style="width:100%;">
</p>

## Dataset
<p align="left">
We obtained the dataset in kaggle. We augmented the dataset using Roboflow and code augmentation to increase its diversity. The dataset was then divided into training set, validation set and testing set. The training set is used to train the DenseNet model, the validation set is used to validate the model during training, while the testing set is used to evaluate its performance.
</p>

| Labels    | Description   |
|------------|------------|
| Mature | <i>In mature cataracts, the lens of the eye becomes completely clouded, causing significant or total vision loss. At this stage, the entire lens is white or gray and the patient may only be able to distinguish light from darkness.</i> | 
| Immature | <i>In immature cataracts, the lens of the eye starts to cloud but not completely. Vision begins to be impaired but there are still parts of the lens that are clear, allowing some light to pass through the lens.</i> | 
| Normal | <i>Normal eyes have clear and transparent lenses, allowing light to pass unhindered onto the retina, which results in clear vision.</i> | 

## Model Architecture
<p align="left">
The DenseNet architecture, or Densely Connected Convolutional Networks, is a type of convolutional neural network that introduces dense connectivity between its layers. We chose it for the cataract classification task for various technical and practical reasons that make it superior in detecting and classifying medical images with high precision.
 </p>

 ## Training
Train DenseNet models using image datasets that are already labeled in the training set. Utilize reputable machine learning frameworks or libraries such as TensorFlow or Keras that provide ready-to-use DenseNet implementations. Adjust hyperparameters and training configurations based on experimentation and evaluation of model performance to obtain optimal results.

### Fine-tune hyperparameters and Training configurations
| Type    | Value    |
|------------|------------|
| Learning Rate | <code>0.0001</code> | 
| Optimizer | <code>RMSprop</code> | 
| Batch Size | <code>16</code> | 
| Number of Training Epochs | <code>20</code> | 
| Input Shape | <code>(416,416,3)</code> | 
| Regularization Techniques |  <code>layers.GlobalMaxPooling2D()(base_model.output)</code><br>layers.Dense(512, activation='relu')(x)</code><br><code>layers.Dropout(0.15)(x)</code><br><code>layers.Dense(3, activation='softmax')(x)
</code><br> | 

### Model Accuracy & Lose
<code>- loss: 0.0931 - accuracy: 0.9635 - val_loss: 0.0503 - val_accuracy: 0.9841</code>
<p align="left">
  <img src="https://github.com/CAPSTONE-C241-PS078/Machine-Learning/blob/5ca777939c3bc31231e08926a78d546da219ea9a/Result/Accuration%20and%20Loss.png" alt="Deskripsi Gambar" style="width:50%; border: 1px solid black;">
</p>

### Evaluate Model
<code>- Test Accuracy: 0.9844 - Test Loss: 0.0542</code>
<p align="left">

### Classification Report at Test Dataset
<p align="left">
  <img src="https://github.com/CAPSTONE-C241-PS078/Machine-Learning/blob/5ca777939c3bc31231e08926a78d546da219ea9a/Result/Classification%20Report.png" alt="Deskripsi Gambar" style="width:50%; border: 1px solid black;">
</p>

### Confusion Matrix at Test Dataset
<p align="left">
  <img src="https://github.com/CAPSTONE-C241-PS078/Machine-Learning/blob/5ca777939c3bc31231e08926a78d546da219ea9a/Result/Confusion%20Matrix.png" alt="Deskripsi Gambar" style="width:50%; border: 1px solid black;">
</p>

## Example Prediction
<p align="left">
  <img src="https://github.com/CAPSTONE-C241-PS078/Machine-Learning/blob/5ca777939c3bc31231e08926a78d546da219ea9a/Result/Predict%20image%20(1).png" alt="Deskripsi Gambar" style="width:100%; border: 1px solid black;">
</p>

