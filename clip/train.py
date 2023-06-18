import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
from clip import CLIP
from dataset import UnsplashData, Flickr8k, Flick8kTFDataset
from tensorflow.keras.utils import Progbar

DEFAULT_TRAIN_CONFIG = {
    "epochs": 20,
    "train_size": 0.8,
    "batch_size": 128,
    "max_samples": 40000,
    "learning_rate": 1E-3,
    "caption_max_length": 200,
    "embedding_size": 256,
    "enable_wand": True
}


class CLIPTrainer:
    def __init__(self, train_config=DEFAULT_TRAIN_CONFIG) -> None:
        super().__init__()
        self.train_config = train_config

        self.text_tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = TFBertForSequenceClassification(BertConfig()).bert
        self.image_encoder = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            pooling='avg'
        )

        # self.dataset = UnsplashData(
        #     f"/Users/vinayakthapliyal/Projects/datasets/unsplash-research-dataset-lite-latest",
        #     self.text_tokenizer,
        #     self.train_config["caption_max_length"]
        # )
        # self.dataset = Flickr8k(
        #     f"/Users/vinayakthapliyal/Projects/datasets/flickr_8k",
        #     self.text_tokenizer,
        #     self.train_config["caption_max_length"],
        #     self.train_config["train_size"]
        # )
        self.dataset = Flick8kTFDataset(
            f"/Users/vinayakthapliyal/Projects/datasets/flickr_8k",
            self.text_tokenizer,
            self.train_config["caption_max_length"],
            self.train_config["train_size"]
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.train_config["learning_rate"]) #,clipnorm=2.0,beta_1=0.99)
        self.model = CLIP(self.image_encoder,
                          self.text_tokenizer,
                          self.text_encoder,
                          self.train_config["embedding_size"],
                          self.train_config["embedding_size"])
        self.model.compile(optimizer=self.optimizer)

        if self.train_config["enable_wand"]:
            wandb.init(
                project="clip",
                config=train_config,
                sync_tensorboard=True
            )

    # @tf.function
    # def train_step(self, model, optimizer, images, captions, captions_mask):
    #     with tf.GradientTape() as tape:
    #         text_embeddings, image_embeddings = model([images, captions, captions_mask])
    #         loss = model.compute_loss(text_embeddings, image_embeddings)
    #     grads = tape.gradient(loss, model.trainable_variables)
    #
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     m = model.trainable_variables
    #     return loss, grads, m
    #
    # @tf.function
    # def test_step(self, model, images, captions, captions_mask):
    #     text_embeddings, image_embeddings = model([images, captions, captions_mask])
    #     loss = model.compute_loss(text_embeddings, image_embeddings)
    #     return loss
    #
    # # @tf.function
    # def train(self):
    #     metrics_names = ['loss']
    #     best_loss = float('inf')
    #
    #     for epoch in range(self.train_config["epochs"], ):
    #         train_dataset, test_dataset, _, _ = self.dataset.get_as_tf_dataset(
    #             self.train_config["batch_size"],
    #             self.train_config["max_samples"]
    #         )
    #         print(f"Epoch {epoch}")
    #         print("Training..")
    #         progress_bar = Progbar(len(train_dataset), stateful_metrics=metrics_names)
    #
    #         for step, (images_batch_train, captions_batch_train, captions_mask_batch_train) in enumerate(train_dataset):
    #             current_loss, grads, m = self.train_step(self.model, self.optimizer, images_batch_train, captions_batch_train, captions_mask_batch_train)
    #             avgs = []
    #             zeros = 0
    #             zero_percentages = 0
    #             nones = 0
    #             counts = 0
    #             for grad in grads:
    #                 if grad is not None:
    #                     np_grad = tf.convert_to_tensor(grad).numpy()
    #                     counts += np.size(np_grad)
    #                     avgs += [np.mean(np_grad)]
    #                     zeros += np.count_nonzero(np_grad == 0.0)
    #                     zero_percentages += np.count_nonzero(np_grad == 0.0)/np_grad.size
    #                 else:
    #                     nones += 1
    #             avg = np.mean(avgs)
    #             zero_percentage = np.mean(zero_percentages)
    #             print()
    #             print(f"Num grads = {counts}")
    #             print(f"Avg grad = {avg}")
    #             print(f"Zero grads = {zeros}")
    #             print(f"Zero grad percentage = {zero_percentage}")
    #             print(f"None grads = {nones}")
    #
    #             progress_bar.update(step + 1, values=[('loss',current_loss)])
    #             if self.train_config["enable_wand"]:
    #                 wandb.log({"train_loss": current_loss,
    #                            "train_epoc": epoch,
    #                            "train_step": step,
    #                            "train_avg_grads": avg,
    #                            "train_zero_grads": zeros,
    #                            "train_zero_grads_perc": zero_percentage,
    #                            "train_none_grads": nones
    #                            })
    #
    #         print("Testing..")
    #         progress_bar = Progbar(len(test_dataset), stateful_metrics=metrics_names)
    #         test_losses = []
    #         for step, (images_batch_test, captions_batch_test, captions_mask_batch_test) in enumerate(test_dataset):
    #             current_loss = self.test_step(self.model, images_batch_test, captions_batch_test, captions_mask_batch_test)
    #             if self.train_config["enable_wand"]:
    #                 wandb.log({"test_loss": current_loss,
    #                            "test_epoc": epoch,
    #                            "test_step": step
    #                            })
    #             test_losses += [current_loss]
    #             progress_bar.update(step + 1, values=[('loss', current_loss)])
    #         test_loss = np.mean(test_losses)
    #         self.model.save(f"clip_model_epoch_{epoch}")
    #         if test_loss < best_loss:
    #             print(f"Saving best model at {epoch}")
    #             best_loss = test_loss
    #             self.model.save("clip_model_best")


    def train_wrapped(self):
        train_dataset, test_dataset, _ = self.dataset.get_as_tf_dataset(
            self.train_config["batch_size"]
            # self.train_config["max_samples"]
        )
        for step, (images_batch_train, captions_batch_train, captions_mask_batch_train) in enumerate(train_dataset):
            imgs = captions_batch_train.numpy()
            print(self.text_tokenizer.decode(imgs[0]))
            if step > 2:
                break
        # Create a learning rate scheduler callback.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3
        )
        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        callbacks = [reduce_lr,
                       early_stopping,
                       tf.keras.callbacks.TensorBoard(update_freq=5)
                       ]
        if self.train_config["enable_wand"]:
            callbacks += [WandbCallback(log_batch_frequency=5)]
        history = self.model.fit(
            train_dataset,
            epochs=self.train_config["epochs"],
            validation_data=test_dataset,
            callbacks=callbacks,
        )
        print("Training completed. Saving model")
        self.model.save("best_clip")
        print("Models are saved.")

trainer = CLIPTrainer(DEFAULT_TRAIN_CONFIG)
trainer.train_wrapped()
# trainer.train()