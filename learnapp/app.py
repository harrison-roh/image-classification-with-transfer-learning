import os
import yaml

from flask import Flask
from flask import request, jsonify

import tensorflow as tf
import tensorflow_datasets as tfds

app = Flask(__name__)


IMAGE_SIZE = 224

@app.route("/model/<model_name>", methods=["POST"])
def create_model(model_name):
    if model_name == "":
        return error_response(400, "Invalid model name")

    params = request.get_json()

    s, ok = check_necessary_params(params)
    if not ok:
        return error_response(400, s)

    tag = params.get("tag", "")
    trial = params.get("trial", False)

    if tag != "" or trial:
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

    labels_file = "lables"
    labels_path = tf.keras.utils.get_file(
        labels_file,
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
        cache_subdir="",
        cache_dir=model_path,
    )

    # signature는 함수를 구분하며, 기본 함수 signature를 이용
    input_name = f"{tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY}_{model.input_names[0]}"
    output_name = "StatefulPartitionedCall"

    desc = params.get("desc")

    cfg = {
        "name": model_name,
        "type": "base",
        # meta graph를 명시하며 "serving"을 사용
        "tags": [tf.saved_model.SERVING],
        # ignore batch size
        "input_shape": list(model.input_shape[1:]),
        "input_operation_name": input_name,
        "output_operation_name": output_name,
        "labels_file": labels_file,
        "description": desc,
    }

    cfg_file = params.get("configFile")
    with open(os.path.join(model_path, cfg_file), "w") as fp:
        yaml.dump(cfg, fp)

    return jsonify({
        "modelName": model_name,
        "modelPath": model_path,
        "modelType": "base",
        "configFilePath": os.path.join(model_path, cfg_file),
        "lablesFilePath": os.path.join(model_path, labels_file),
    })


def create_transfer_learned_model(model_name, params):
    trial = params.get("trial", False)

    base_model = get_base_model(True)
    if trial:
        model, result = trial_trasnfer_learned_model(base_model, params)
    else:
        return error_response(500, "Not yet implemented: transfer learning")

    model_path = params.get("modelPath")
    model.save(model_path)

    labels_file = "lables"
    with open(os.path.join(model_path, labels_file), "w") as fp:
        fp.write("cat\ndog\n")

    # signature는 함수를 구분하며, 기본 함수 signature를 이용
    input_name = f"{tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY}_{model.input_names[0]}"
    output_name = "StatefulPartitionedCall"

    desc = params.get("desc")

    cfg = {
        "name": model_name,
        "type": "trial",
        # meta graph를 명시하며 "serving"을 사용
        "tags": [tf.saved_model.SERVING],
        # ignore batch size
        "input_shape": list(model.input_shape[1:]),
        "input_operation_name": input_name,
        "output_operation_name": output_name,
        "labels_file": labels_file,
        "description": desc,
    }

    cfg_file = params.get("configFile")
    with open(os.path.join(model_path, cfg_file), "w") as fp:
        yaml.dump(cfg, fp)

    result["modelName"] = model_name
    result["modelPath"] = model_path
    result["modelType"] = "trial"
    result["configFilePath"] = os.path.join(model_path, cfg_file)
    result["labelsFilePath"] = os.path.join(model_path, labels_file)

    return jsonify(result)


def trial_trasnfer_learned_model(base_model, params):
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        with_info=True,
        as_supervised=True,
    )

    train = raw_train.map(transform_format)
    validation = raw_validation.map(transform_format)
    test = raw_test.map(transform_format)

    train_batches = train.shuffle(1000).batch(32)
    validation_batches = validation.shuffle(1000).batch(32)
    test_batches = test.shuffle(1000).batch(32)

    for image_batch, label_batch in train_batches.take(1):
        feature_batch = base_model(image_batch)

    # 이지미에서 featur를 추출하는 CNN 모델의 가중치는 조정하지 않고,
    # 분류를 수행하는 fully connected layer만 학습
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    # cats, dogs를 분류하기위한 binary classfication layer 추가
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)

    model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    epochs = params.get("epochs", 1)

    loss0, acc0 = model.evaluate(validation_batches, steps=20)
    history = model.fit(train_batches, epochs=epochs, validation_data=validation_batches)

    loss = history.history["loss"]
    acc = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]

    result = {
        "initLoss": loss0,
        "initAccuracy": acc0,
        "trainLoss": loss,
        "trainAccuracy": acc,
        "validationLoss": val_loss,
        "validationAccuracy": val_acc,
    }

    return model, result


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
