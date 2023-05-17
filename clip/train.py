import pandas as pd
import numpy as np
import wandb
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from clip import CLIP
from dataset import UnsplashData
from tensorflow.keras.utils import Progbar

EPOCHS = 10
TRAIN_SIZE = 0.8
BATCH_SIZE = 128

CAPTION_MAX_LENGTH = 150
EMBEDDING_SIZE = 150

ENABLE_WAND = True

# Modules
text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_encoder = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
image_encoder = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    pooling='avg'
)

# Data
dataset = UnsplashData(
    f"/Users/vinayakthapliyal/Projects/datasets/unsplash-research-dataset-lite-latest/",
    text_tokenizer,
    CAPTION_MAX_LENGTH
)

if ENABLE_WAND:
    wandb.init(
        project="clip",

        config={
            "text_encoder": "distilbert-base-uncased",
            "image_encoder": "tf.keras.applications.resnet50.ResNet50",
            "dataset": dataset,
            "epochs": EPOCHS,
        }
    )
train_dataset, test_dataset = dataset.get_as_tf_dataset(TRAIN_SIZE, BATCH_SIZE, max_items=None)

#Training
optimizer = tf.keras.optimizers.Adam()
model = CLIP(image_encoder, text_tokenizer, text_encoder, EMBEDDING_SIZE, EMBEDDING_SIZE)

@tf.function
def train_step(model, optimizer, images, captions, captions_mask):
    with tf.GradientTape() as tape:
        model([images, captions, captions_mask])
        loss_value = model.losses
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    avg_loss_value = tf.math.reduce_mean(loss_value)
    return avg_loss_value

metrics_names = ['loss']
num_training_samples = len(train_dataset)
best_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")
    current_loss = float('inf')
    progress_bar = Progbar(num_training_samples, stateful_metrics=metrics_names)

    for step, (images_batch_train, captions_batch_train, captions_mask_batch_train) in enumerate(train_dataset):
        current_loss = train_step(model, optimizer, images_batch_train, captions_batch_train, captions_mask_batch_train)
        progress_bar.update(step + 1, values=[('loss',current_loss)])

    if current_loss < best_loss:
        best_loss = current_loss
        # model.save("clip_model")
        print(f"Saved Best Model! at epoch {epoch} step {step}")

# wandb.tensorflow.log(tf.summary.merge_all())

