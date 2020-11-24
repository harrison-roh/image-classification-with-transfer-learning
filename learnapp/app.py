import os
import yaml

from flask import Flask
from flask import request, jsonify
import pymysql

import tensorflow as tf
import tensorflow_datasets as tfds

app = Flask(__name__)

MODEL_TYPE_BASE = "base"
MODEL_TYPE_PRACTICAL = "practical"
MODEL_TYPE_TRIAL = "trial"

BINARY_CLASS = "binary"
MULTI_CLASS = "multi"

LABELS_FILE = "lables"

IMAGE_SIZE = 224


@app.route("/model/<model_name>", methods=["POST"])
def create_model(model_name):
    if model_name == "":
        return error_response(400, "Invalid model name")

    params = request.get_json()

    s, ok = check_necessary_params(params)
    if not ok:
        return error_response(400, s)

    image_path = params.get("imagePath", "")
    trial = params.get("trial", False)

    if image_path != "" or trial:
        return create_transfer_learned_model(model_name, params)
    else:
        return create_base_model(model_name, params)


def check_necessary_params(params):
    model_path = params.get("modelPath", "")
    if model_path == "":
        return "Invalid path for model", False

    cfg_file = params.get("configFile", "")
    if cfg_file == "":
        return "Invalid config file name", False

    return "", True


def get_base_model(is_tl):
    if is_tl:
        return tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            weights="imagenet",
            # fully connected layer인 분류계층은 학습을 하기위해 포함하지 않음
            include_top=False,
        )
    else:
        return tf.keras.applications.MobileNetV2(
            weights="imagenet",
        )


def create_base_model(model_name, params):
    model_path = params.get("modelPath")

    model = get_base_model(False)
    model.save(model_path)

    labels_path = tf.keras.utils.get_file(
        LABELS_FILE,
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
        cache_subdir="",
        cache_dir=model_path,
    )

    # signature는 함수를 구분하며, 기본 함수 signature를 이용
    input_name = (
        f"{tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY}_{model.input_names[0]}"
    )
    output_name = "StatefulPartitionedCall"

    desc = params.get("desc")

    cfg = {
        "name": model_name,
        "type": MODEL_TYPE_BASE,
        "tags": [tf.saved_model.SERVING],  # meta graph를 명시하며 "serving"을 사용
        "classification": MULTI_CLASS,
        "input_shape": list(model.input_shape[1:]),  # ignore batch size
        "input_operation_name": input_name,
        "output_operation_name": output_name,
        "labels_file": LABELS_FILE,
        "description": desc,
    }

    cfg_file = params.get("configFile")
    with open(os.path.join(model_path, cfg_file), "w") as fp:
        yaml.dump(cfg, fp)

    return jsonify(
        {
            "modelName": model_name,
            "modelType": MODEL_TYPE_BASE,
        }
    )


def create_transfer_learned_model(model_name, params):
    trial = params.get("trial", False)

    base_model = get_base_model(True)
    if trial:
        model_type = MODEL_TYPE_TRIAL
        model, classification, labels, result = trial_trasnfer_learned_model(
            base_model, params
        )
    else:
        model_type = MODEL_TYPE_PRACTICAL
        model, classification, labels, result = practical_trasnfer_learned_model(
            base_model, params
        )

    model_path = params.get("modelPath")
    model.save(model_path)

    with open(os.path.join(model_path, LABELS_FILE), "w") as fp:
        for label in labels:
            fp.write(f"{label}\n")

    # signature는 함수를 구분하며, 기본 함수 signature를 이용
    input_name = (
        f"{tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY}_{model.input_names[0]}"
    )
    output_name = "StatefulPartitionedCall"

    desc = params.get("desc")

    cfg = {
        "name": model_name,
        "type": model_type,
        "tags": [tf.saved_model.SERVING],  # meta graph를 명시하며 "serving"을 사용
        "classification": classification,
        "input_shape": list(model.input_shape[1:]),  # ignore batch size
        "input_operation_name": input_name,
        "output_operation_name": output_name,
        "labels_file": LABELS_FILE,
        "description": desc,
    }

    cfg_file = params.get("configFile")
    with open(os.path.join(model_path, cfg_file), "w") as fp:
        yaml.dump(cfg, fp)

    result["modelName"] = model_name
    result["modelType"] = model_type

    return jsonify(result)


def practical_trasnfer_learned_model(base_model, params):
    image_path = params.get("imagePath", "")

    dirs = []
    for file in os.listdir(image_path):
        path = os.path.join(image_path, file)
        if os.path.isdir(path):
            dirs.append(path)

    label_mode = "binary" if len(dirs) == 2 else "categorical"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_path,
        label_mode=label_mode,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_path,
        label_mode=label_mode,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    labels = train_ds.class_names

    train = train_ds.map(transform_format)
    validation = validation_ds.map(transform_format)

    model, classification = build_and_compile_model(base_model, train, len(labels))

    epochs = params.get("epochs", 10)
    result = train_and_evaluate_model(model, train, validation, epochs)

    return model, classification, labels, result


def trial_trasnfer_learned_model(base_model, params):
    (raw_train, raw_validation), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        with_info=True,
        as_supervised=True,
    )

    labels = []
    get_label_name = metadata.features["label"].int2str
    for i in range(metadata.features["label"].num_classes):
        labels.append(get_label_name(i))

    train = raw_train.map(transform_format)
    validation = raw_validation.map(transform_format)

    train_batches = train.shuffle(1000).batch(32)
    validation_batches = validation.shuffle(1000).batch(32)

    model, classification = build_and_compile_model(
        base_model,
        train_batches,
        len(labels),
    )

    epochs = params.get("epochs", 10)
    result = train_and_evaluate_model(model, train_batches, validation_batches, epochs)

    return model, classification, labels, result


def build_and_compile_model(
    base_model, train_batches, nr_classes, lr=0.0001, metrics=["accuracy"]
):
    for image_batch, label_batch in train_batches.take(1):
        feature_batch = base_model(image_batch)

    # 이지미에서 featur를 추출하는 CNN 모델의 가중치는 조정하지 않고,
    # 분류를 수행하는 fully connected layer만 학습
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    # 분류계층에 sigmoid or softmax 활성함수를 적용하기 때문에 from_logits을 False로 해야
    # loss에 대한 학습이 됨 (loss 함수 내부에서는 logit 값을 사용함)
    if nr_classes == 2:
        classification = BINARY_CLASS
        activation = "sigmoid"
        units = 1
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:  # nr_classes > 2
        classification = MULTI_CLASS
        activation = "softmax"
        units = nr_classes
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # nr_classes에 맞춰 분류하기위한 classfication layer 추가
    prediction_layer = tf.keras.layers.Dense(units, activation=activation)
    prediction_batch = prediction_layer(feature_batch_average)

    model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=lr),
        loss=loss,
        metrics=metrics,
    )

    return model, classification


def train_and_evaluate_model(model, train_batches, validation_batches, epochs):
    loss0, acc0 = model.evaluate(validation_batches, steps=20)
    history = model.fit(
        train_batches, epochs=epochs, validation_data=validation_batches
    )

    loss = history.history["loss"]
    acc = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]

    result = {
        "epoches": epochs,
        "initLoss": loss0,
        "initAccuracy": acc0,
        "trainLoss": loss,
        "trainAccuracy": acc,
        "validationLoss": val_loss,
        "validationAccuracy": val_acc,
    }

    return result


def transform_format(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image, label


def error_response(status, message):
    response = jsonify(
        {
            "message": message,
        }
    )
    response.status_code = status

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="18090", debug=True)
