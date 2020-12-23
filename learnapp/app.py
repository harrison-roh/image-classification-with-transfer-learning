import os
import yaml
import time
import requests
import queue
import errno
import threading
import multiprocessing as mp

from flask import Flask
from flask import request, jsonify

import tensorflow as tf
import tensorflow_datasets as tfds

CLSAPP = "app:18080"
running = True

MAX_CONCURRENT = 4

app = Flask(__name__)
q = queue.Queue(maxsize=10)

MODEL_TYPE_BASE = "base"
MODEL_TYPE_PRACTICAL = "practical"
MODEL_TYPE_TRIAL = "trial"

BINARY_CLASS = "binary"
MULTI_CLASS = "multi"

LABELS_FILE = "lables"

TRAINING_EPOCHS_DEFAULT = 10
IMAGE_SIZE = 224


class DeferredDelDict(dict):
    _dels = None

    def __enter__(self):
        self._dels = set()

    def __exit__(self, type, value, traceback):
        for key in self._dels:
            try:
                dict.__delitem__(self, key)
            except KeyError:
                pass
        self._dels = None

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(str(key))

        dict.__delitem__(self, key) if self._dels is None else self._dels.add(key)


tasks = DeferredDelDict()


class ModelRequest:
    def __init__(self, model_name, model_type, params):
        self.model_name = model_name
        self.model_type = model_type
        self.params = params


def shutdown_server():
    stop = request.environ.get("werkzeug.server.shutdown")
    if stop is None:
        raise RuntimeError("Not running with the Werkzeug Server")

    print("Shutting down...")
    stop()


@app.route("/shutdown", methods=["POST"])
def shutdown():
    global running
    running = False

    remaining_requests = q.qsize()
    building_requests = len(tasks)

    shutdown_server()

    return jsonify(
        {
            "remainingRequests": remaining_requests,
            "buildingRequests": building_requests,
        }
    )


@app.route("/models", methods=["GET"])
def get_models():
    remaining_requests = q.qsize()
    building_requests = len(tasks)

    return jsonify(
        {
            "remainingRequests": remaining_requests,
            "buildingRequests": building_requests,
        }
    )


@app.route("/models/<model_name>", methods=["POST"])
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
        if trial:
            model_type = MODEL_TYPE_TRIAL
        else:
            model_type = MODEL_TYPE_PRACTICAL
    else:
        model_type = MODEL_TYPE_BASE

    req = ModelRequest(model_name, model_type, params)
    try:
        q.put_nowait(req)
    except queue.Full:
        return error_response(500, "Server currently busy")

    return jsonify(
        {
            "model": model_name,
            "type": model_type,
        }
    )


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
    model = get_base_model(False)

    model_path = params.get("modelPath")
    if os.path.isdir(model_path):
        print(f"Model path already exists: {model_path}")
        return

    model.save(model_path)

    tmp_labels_file = f"{LABELS_FILE}.tmp"

    labels_path = tf.keras.utils.get_file(
        tmp_labels_file,
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
        cache_subdir="",
        cache_dir=model_path,
    )
    # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a#gistcomment-2719675
    # 불필요한 첫번째 label(`background`)를 제거
    with open(os.path.join(model_path, LABELS_FILE), "w") as ofp:
        with open(os.path.join(model_path, tmp_labels_file)) as ifp:
            labels = ifp.readlines()
        os.remove(os.path.join(model_path, tmp_labels_file))
        for label in labels[1:]:
            ofp.write(f"{label}")

    # signature는 함수를 구분하며, 기본 함수 signature를 이용
    input_name = (
        f"{tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY}_{model.input_names[0]}"
    )
    output_name = "StatefulPartitionedCall"

    desc = params.get("desc", "")
    if desc == "":
        desc = "Default base model"

    cfg = {
        "name": model_name,
        "type": MODEL_TYPE_BASE,
        "tags": [tf.saved_model.SERVING],  # meta graph를 명시하며 "serving"을 사용
        "classification": MULTI_CLASS,
        "inputShape": list(model.input_shape[1:]),  # ignore batch size
        "inputOperationName": input_name,
        "outputOperationName": output_name,
        "labelsFile": LABELS_FILE,
        "description": desc,
    }

    cfg_file = params.get("configFile")
    with open(os.path.join(model_path, cfg_file), "w") as fp:
        yaml.dump(cfg, fp)

    response = requests.put(
        f"http://{CLSAPP}/models/{model_name}", json={"modelPath": model_path}
    )
    print(
        f"Operate {model_name}, {MODEL_TYPE_BASE}, {model_path}: ({response.status_code}) {response.text}"
    )


def create_transfer_learned_model(model_name, params):
    trial = params.get("trial", False)
    epochs = params.get("epochs", TRAINING_EPOCHS_DEFAULT)

    base_model = get_base_model(True)
    if trial:
        model_type = MODEL_TYPE_TRIAL
        model, classification, labels, result = trial_trasnfer_learned_model(
            base_model, epochs
        )
    else:
        model_type = MODEL_TYPE_PRACTICAL
        image_path = params.get("imagePath", "")
        model, classification, labels, result = practical_trasnfer_learned_model(
            base_model, image_path, epochs
        )

    model_path = params.get("modelPath")
    if os.path.isdir(model_path):
        print(f"Model path already exists: {model_path}")
        return

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
        "inputShape": list(model.input_shape[1:]),  # ignore batch size
        "inputOperationName": input_name,
        "outputOperationName": output_name,
        "labelsFile": LABELS_FILE,
        "description": desc,
        "trainingResult": result,  # 학습결과 저장
    }

    cfg_file = params.get("configFile")
    with open(os.path.join(model_path, cfg_file), "w") as fp:
        yaml.dump(cfg, fp)

    response = requests.put(
        f"http://{CLSAPP}/models/{model_name}", json={"modelPath": model_path}
    )
    print(
        f"Operate {model_name}, {MODEL_TYPE_BASE}, {model_path}: ({response.status_code}) {response.text}"
    )


def practical_trasnfer_learned_model(base_model, image_path, epochs):
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

    train = train_ds.map(normalize_image)
    validation = validation_ds.map(normalize_image)

    model, classification = build_and_compile_model(base_model, train, len(labels))

    result = train_and_evaluate_model(model, train, validation, epochs)

    return model, classification, labels, result


def trial_trasnfer_learned_model(base_model, epochs):
    (raw_train, raw_validation), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:30%]", "train[80%:]"],
        with_info=True,
        as_supervised=True,
    )

    labels = []
    get_label_name = metadata.features["label"].int2str
    for i in range(metadata.features["label"].num_classes):
        labels.append(get_label_name(i))

    train = raw_train.map(normalize_and_resize_image)
    validation = raw_validation.map(normalize_and_resize_image)

    train_batches = train.shuffle(1000).batch(32)
    validation_batches = validation.shuffle(1000).batch(32)

    model, classification = build_and_compile_model(
        base_model,
        train_batches,
        len(labels),
    )

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
        "epochs": epochs,
        "initLoss": loss0,
        "initAccuracy": acc0,
        "trainLoss": loss,
        "trainAccuracy": acc,
        "validationLoss": val_loss,
        "validationAccuracy": val_acc,
    }

    return result


def normalize_image(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image, label


def normalize_and_resize_image(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image, label


def error_response(status, message):
    response = jsonify(
        {
            "error": message,
        }
    )
    response.status_code = status

    return response


def overwatch_tasks(tasks, timeout=None):
    with tasks:
        for task in tasks:
            try:
                task.get(timeout)
            except mp.TimeoutError:
                continue

            del tasks[task]


def management(nr_workers=MAX_CONCURRENT):
    global tasks
    with mp.Pool(processes=nr_workers) as pool:
        while running:
            overwatch_tasks(tasks, 1)
            if len(tasks) >= MAX_CONCURRENT or q.empty():
                time.sleep(1)
                continue

            while running and len(tasks) < MAX_CONCURRENT:
                try:
                    req = q.get_nowait()
                except queue.Empty:
                    break

                if (
                    req.model_type == MODEL_TYPE_PRACTICAL
                    or req.model_type == MODEL_TYPE_TRIAL
                ):
                    task = pool.apply_async(
                        func=create_transfer_learned_model,
                        args=(
                            req.model_name,
                            req.params,
                        ),
                    )
                else:
                    task = pool.apply_async(
                        func=create_base_model,
                        args=(
                            req.model_name,
                            req.params,
                        ),
                    )

                tasks[task] = True

        overwatch_tasks(tasks)

    print("Exit manager")


if __name__ == "__main__":
    manager = threading.Thread(target=management)
    manager.start()

    app.run(host="0.0.0.0", port="18090", debug=True, use_reloader=False)

    manager.join()
