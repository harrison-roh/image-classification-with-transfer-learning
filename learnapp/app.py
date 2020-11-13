import os
import yaml

from flask import Flask
from flask import request, jsonify

import tensorflow as tf

app = Flask(__name__)


@app.route("/model/<model_name>", methods=["POST"])
def create_model(model_name):
    if model_name == "":
        return error_response(400, "Invalid model name")

    params = request.get_json()

    model_path = params.get("modelPath", "")
    if model_path == "":
        return error_response(400, "Invalid path for model")

    cfg_file = params.get("configFile", "")
    if cfg_file == "":
        return error_response(400, "Invalid config file name")

    desc = params.get("desc", "")

    model = tf.keras.applications.MobileNetV2(
        include_top=True,
        weights="imagenet",
    )
    model.save(model_path)

    labels_file = "lables"
    labels_path = tf.keras.utils.get_file(
        labels_file,
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
        cache_subdir="",
        cache_dir=model_path,
    )

    cfg = {
        "name": model_name,
        # meta graph를 명시하며 "serving"을 사용
        "tags": [tf.saved_model.SERVING],
        # ignore batch size
        "input_shape": list(model.input_shape[1:]),
        # signature는 함수를 구분하며, 기본 함수 signature를 이용
        "input_operation_name": f"{tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY}_{model.input_names[0]}",  # TODO
        "output_operation_name": "StatefulPartitionedCall",  # TODO
        "labels_file": labels_file,
        "description": desc,
    }

    with open(os.path.join(model_path, cfg_file), "w") as fp:
        yaml.dump(cfg, fp)

    response = {
        "modelName": model_name,
        "modelPath": model_path,
        "configFilePath": os.path.join(model_path, cfg_file),
        "lablesFilePath": os.path.join(model_path, labels_file),
    }

    return jsonify(response)


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
