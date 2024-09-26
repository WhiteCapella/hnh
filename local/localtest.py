from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import DefaultDataCollator
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import DefaultDataCollator
import evaluate
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer



def main():
    food = load_dataset("food101", split="train[:5000]")
    food = food.train_test_split(test_size=0.2)

    food["train"][0]

    labels = food["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    id2label[str(79)]

    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    food = food.with_transform(transforms)
    data_collator = DefaultDataCollator()

    size = (image_processor.size["height"], image_processor.size["width"])
    train_data_augmentation = keras.Sequential(
       [
            layers.RandomCrop(size[0], size[1]),
            layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="train_data_augmentation",
    )
    val_data_augmentation = keras.Sequential(
       [
            layers.CenterCrop(size[0], size[1]),
            layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        ],
        name="val_data_augmentation",
    )
    
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )   

    training_args = TrainingArguments(
        output_dir="my_awesome_food_model",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=food["train"],
        eval_dataset=food["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()



    ds = load_dataset("food101", split="validation[:10]")
    image = ds["image"][0]

    classifier = pipeline("image-classification", model="my_awesome_food_model")
    classifier(image)

    image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
    inputs = image_processor(image, return_tensors="pt")

    model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    model.config.id2label[predicted_label]
    'beignets'

    image_processor = AutoImageProcessor.from_pretrained("MariaK/food_classifier")
    inputs = image_processor(image, return_tensors="tf")

    model = TFAutoModelForImageClassification.from_pretrained("MariaK/food_classifier")
    logits = model(**inputs).logits

    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    model.config.id2label[predicted_class_id]
    'beignets'


def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # `expand_dims()` is used to add a batch dimension since
    # the TF augmentation layers operates on batched inputs.
    return tf.expand_dims(tf_image, 0)

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

main()
