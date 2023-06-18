import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

BUFFER_SIZE=1000

class Flick8kTFDataset:
    def __init__(self, dir, text_tokenizer, text_max_length=150, train_size=0.8) -> None:
        super().__init__()
        self.dir = dir
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.train_size = train_size

    def tokenize_text(self, text):
        tokenized_text = self.text_tokenizer(
            tf.compat.as_str(text.numpy()),
            add_special_tokens = True,
            padding = "max_length",
            max_length = self.text_max_length,
            return_tensors = "tf"
            )
        return tokenized_text["input_ids"][0], tokenized_text["attention_mask"][0]

    def process_row(self, metadata_row):
        image_path = metadata_row["photo_path"]
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        processed_img = preprocess_input(img)
        captions, masks = tf.py_function(self.tokenize_text,
                                              inp=[metadata_row["caption"]],
                                              Tout=(tf.int32, tf.int32))

        captions.set_shape([self.text_max_length])
        masks.set_shape([self.text_max_length])

        return processed_img, captions, masks

    def get_as_tf_dataset(self, batch_size=32, max_items=None):
        photos_path = self.dir + "/Images/"
        metadata = pd.read_csv(self.dir + "/captions.txt")
        metadata["photo_path"] = photos_path + metadata.image

        if max_items is None:
            max_items = len(metadata)

        train_size = int(max_items * self.train_size)
        dataset = tf.data.Dataset.from_tensor_slices(dict(metadata))
        full_dataset = dataset.shuffle(BUFFER_SIZE)

        dataset_with_images = full_dataset.map(self.process_row, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_dataset = dataset_with_images.take(train_size).batch(batch_size)
        test_dataset = dataset_with_images.skip(train_size).batch(batch_size)
        return train_dataset, test_dataset, metadata


class Dataset:
    def __init__(self, dir, text_tokenizer, text_max_length=150, train_size=0.8) -> None:
        super().__init__()
        self.dir = dir
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.metadata = self.get_images_and_captions()
        self.train_size = train_size
        num_samples = len(self.metadata)
        num_train_samples = int(num_samples * train_size)
        self.train_metadata = self.metadata[0:num_train_samples]
        self.test_metadata = self.metadata[num_train_samples:]

    def get_encoded_images_and_captions(self, image_paths_and_captions):
        num_samples = len(image_paths_and_captions)
        images = np.zeros(shape=(num_samples, 224, 224, 3))
        captions = np.zeros(shape=(num_samples, self.text_max_length), dtype=np.int32)
        captions_mask = np.zeros(shape=(num_samples, self.text_max_length), dtype=np.int32)

        print("Loading images and tokenizing text")
        for index, photo_description in tqdm(image_paths_and_captions.iterrows(), total=num_samples):
            photo_path = photo_description.photo_path
            img = image.load_img(photo_path, target_size=(224, 224))
            batched_img = np.expand_dims(img, axis=0)
            processed_img = preprocess_input(batched_img)
            images[index] = processed_img
            tokenized_text = self.text_tokenizer(
                photo_description.caption,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.text_max_length,
                return_tensors="tf"
            )
            captions[index] = tokenized_text["input_ids"]
            captions_mask[index] = tokenized_text["attention_mask"]
        return images, captions, captions_mask

    def get_as_tf_dataset(self, batch_size=32, max_items=None):

        # shuffle captions
        train_metadata_full_shuffled = self.train_metadata.sample(frac=1).reset_index(drop=True)
        test_metadata_full_shuffled = self.test_metadata.sample(frac=1).reset_index(drop=True)

        num_train_samples = int(max_items * self.train_size)
        train_metadata = train_metadata_full_shuffled[0:max_items]
        test_metadata = test_metadata_full_shuffled[0:(max_items-num_train_samples)]

        train_images, train_captions, train_captions_mask = self.get_encoded_images_and_captions(train_metadata)
        test_images, test_captions, test_captions_mask = self.get_encoded_images_and_captions(test_metadata)

        tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_captions, train_captions_mask))
        tf_test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_captions, test_captions_mask))

        tf_train_dataset = tf_train_dataset.batch(batch_size)
        tf_test_dataset = tf_test_dataset.batch(batch_size)
        return tf_train_dataset, tf_test_dataset, train_metadata#, test_metadata

    def get_images_and_captions(self):
        raise NotImplementedError("Implement this pls")


class Flickr8k(Dataset):

    def get_images_and_captions(self) -> pd.DataFrame:
        photos_path = self.dir + "/Images/"
        photos_metadata = pd.read_csv(self.dir + "/captions.txt")

        result = photos_metadata[["image", "caption"]]
        result["photo_path"] = photos_path + result.image
        return result


class UnsplashData(Dataset):

    def get_images_and_captions(self) -> pd.DataFrame:
        photos_path = self.dir + "/photos/"
        photos_metadata_raw = \
            pd.read_csv(self.dir + "/photos.tsv000",
                        sep='\t',
                        header=0)
        photos_metadata = photos_metadata_raw.loc[photos_metadata_raw.ai_description.notna()]
        photos_metadata = photos_metadata.reset_index()
        result = photos_metadata[["photo_id", "ai_description"]]
        result["caption"] = photos_metadata.ai_description
        result["photo_path"] = photos_path + result.photo_id + ".jpg"
        result = result.sample(frac=1).reset_index(drop=True)
        return result
