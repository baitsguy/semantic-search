import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


class UnsplashData:
    BATCH_SIZE = 32

    def __init__(self, dir, text_tokenizer, text_max_length=150) -> None:
        super().__init__()
        self.dir = dir
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length

    def get_as_tf_dataset(self, train_size=0.8, batch_size=32, max_items=None, shuffle=True):
        if max_items is None:
            metadata = self.get_images_and_captions()
        else:
            metadata = self.get_images_and_captions()[0:max_items]
        num_samples = len(metadata)
        num_train_samples = int(num_samples * train_size)
        images = np.zeros(shape=(num_samples, 224, 224, 3))
        captions = np.zeros(shape=(num_samples, self.text_max_length), dtype=np.int32)
        captions_mask = np.zeros(shape=(num_samples, self.text_max_length), dtype=np.int32)

        print("Loading images and tokenizing text")
        for index, photo_description in tqdm(metadata.iterrows(), total=num_samples):
            photo_path = photo_description.photo_path
            img = image.load_img(photo_path, target_size=(224, 224))
            batched_img = np.expand_dims(img, axis=0)
            processed_img = preprocess_input(batched_img)
            images[index] = processed_img
            tokenized_text = self.text_tokenizer(
                photo_description.ai_description,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.text_max_length,
                return_tensors="tf"
            )
            captions[index] = tokenized_text["input_ids"]
            captions_mask[index] = tokenized_text["attention_mask"]
        tf_train_dataset = tf.data.Dataset.from_tensor_slices((images[:num_train_samples], captions[:num_train_samples], captions_mask[:num_train_samples]))
        tf_test_dataset = tf.data.Dataset.from_tensor_slices((images[num_train_samples:], captions[num_train_samples:], captions_mask[num_train_samples:]))
        if shuffle:
            tf_train_dataset = tf_train_dataset.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
            tf_test_dataset = tf_test_dataset.shuffle(buffer_size=num_samples, reshuffle_each_iteration=False)
        tf_train_dataset = tf_train_dataset.batch(batch_size)
        tf_test_dataset = tf_test_dataset.batch(batch_size)
        return tf_train_dataset, tf_test_dataset

    def get_images_and_captions(self) -> pd.DataFrame:
        photos_path = self.dir + "/photos/"
        photos_metadata_raw = \
            pd.read_csv(self.dir + "/photos.tsv000",
                        sep='\t',
                        header=0)
        photos_metadata = photos_metadata_raw.loc[photos_metadata_raw.ai_description.notna()]
        photos_metadata = photos_metadata.reset_index()
        result = photos_metadata[["photo_id", "ai_description"]]
        result["photo_path"] = photos_path + result.photo_id + ".jpg"
        return result
