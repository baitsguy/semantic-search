import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

class Head(tf.keras.layers.Layer):
    def __init__(self, output_dim=256, projection_dim=256):
        super().__init__()
        self.projection = tf.keras.layers.Dense(projection_dim)
        self.activation = tf.keras.layers.Activation(tf.nn.gelu)
        self.dense = tf.keras.layers.Dense(output_dim)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.normalized = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x1 = self.projection(inputs)
        x2 = self.activation(x1)
        x3 = self.dense(x2)
        x4 = self.dropout(x3)
        x5 = x4 + x1
        x6 = self.normalized(x5)
        return x6


class CLIP(tf.keras.Model):

    def __init__(self,
                 image_encoder,
                 text_tokenizer,
                 text_encoder,
                 image_embedding_dim,
                 text_embedding_dim,
                 temperature=1):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self.image_head = Head(image_embedding_dim)
        self.text_head = Head(text_embedding_dim)
        self.temperature = temperature

        self.image_encoder.trainable = False
        self.text_tokenizer.trainable = False
        self.text_encoder.trainable = False

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    # @tf.function
    def call(self, batched_input):
        image_batch = batched_input[0]
        text_input_id_batch = batched_input[1]
        text_mask_batch = batched_input[2]
        image_encodings = self.image_encoder(image_batch)

        text_encoder_hidden_states = self.text_encoder(
            input_ids=text_input_id_batch,
            attention_mask=text_mask_batch,
            # output_hidden_states=True
        ).last_hidden_state

        text_encodings = text_encoder_hidden_states[:, 0, :]

        image_embeddings = self.image_head(image_encodings)
        text_embeddings = self.text_head(text_encodings)

        # image_embeddings = image_embeddings / tf.norm(image_embeddings, axis=-1, keepdims=True)
        # text_embeddings = text_embeddings / tf.norm(text_embeddings, axis=-1, keepdims=True)

        # text_loss = self.contrastive_loss(logits)
        # image_loss = self.contrastive_loss(tf.transpose(logits))

        # self.add_loss((text_loss + image_loss)/ 2.0)
        return text_embeddings, image_embeddings

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_loss(self, text_embeddings, image_embeddings):
        logits = tf.matmul(text_embeddings, image_embeddings, transpose_b=True)

        images_similarity = tf.matmul(image_embeddings, image_embeddings, transpose_b=True)
        captions_similarity = tf.matmul(text_embeddings, text_embeddings, transpose_b=True)

        targets = tf.keras.activations.softmax((captions_similarity + images_similarity) / (2 * self.temperature))

        captions_loss = tf.keras.losses.categorical_crossentropy(y_true=targets, y_pred=logits, from_logits=True)
        images_loss = tf.keras.losses.categorical_crossentropy(y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True)

        loss = (captions_loss + images_loss) / 2
        return loss


    def train_step(self, features):
        with tf.GradientTape() as tape:
            text_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss(text_embeddings, image_embeddings)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        text_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(text_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
