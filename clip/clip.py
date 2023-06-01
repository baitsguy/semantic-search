import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

class Head(tf.keras.layers.Layer):
    def __init__(self, output_dim=256):
        super().__init__()
        self.projection = tf.keras.layers.Dense(output_dim, activation=tf.nn.gelu)
        # self.dense = tf.keras.layers.Dense(output_dim)
        # self.dropout = tf.keras.layers.Dropout(0.1)
        # self.normalized = tf.keras.layers.Normalization()

    def call(self, inputs):
        x1 = self.projection(inputs)
        # x2 = self.dense(x1)
        # x3 = self.dropout(x2)
        # x4 = self.normalized(x3)
        return x1


class CLIP(tf.keras.Model):

    def __init__(self,
                 image_encoder,
                 text_tokenizer,
                 text_encoder,
                 image_embedding_dim,
                 text_embedding_dim):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self.image_head = Head(image_embedding_dim)
        self.text_head = Head(text_embedding_dim)

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

        image_embeddings = image_embeddings / tf.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings = text_embeddings / tf.norm(text_embeddings, axis=-1, keepdims=True)

        logits = tf.matmul(text_embeddings, image_embeddings, transpose_b=True)

        # loss
        # images_similarity = tf.matmul(image_embeddings, tf.transpose(image_embeddings))
        # texts_similarity = tf.matmul(text_embeddings, tf.transpose(text_embeddings))
        # targets = (images_similarity + texts_similarity) / 2.0
        #
        # tf.keras.metrics.sparse_categorical_crossentropy(
        #     y_true=tf.range(logits.shape[0]), y_pred=logits, from_logits=True
        # )
        #
        # text_loss = BinaryCrossentropy(reduction='none', axis=-0)(logits, targets)
        # image_loss = BinaryCrossentropy(reduction='none', axis=1)(logits, targets)

        text_loss = self.contrastive_loss(logits)
        image_loss = self.contrastive_loss(tf.transpose(logits))

        self.add_loss((text_loss + image_loss)/ 2.0)
        return logits

    @tf.function
    def contrastive_loss(self, logits) :
        return tf.math.reduce_mean(
            tf.keras.metrics.sparse_categorical_crossentropy(
                y_true=tf.range(tf.shape(logits)[0]), y_pred=logits, from_logits=True
            )
        )