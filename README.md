## 전이학습을 이용한 이미지 분류

keras를 이용해 전이학습을 통한 모델을 생성하며, GO tensorflow를 이용해 모델 로드 및 이미지를 추론.
전이학습을 위한 사용자 이미지를 업로드 하고, 기본 모델 기반의 전이학습을 통해 분류계층을 학습함으로써 해당 이미지에 특화된 분류 모델을 생성.

### base model

[Pre-trained MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2)

## 시작하기

### 사전작업

OS에 따라 `.env` 파일의 `USER_UID`를 수정

#### USER_UID

- Linux
  - `USER_UID=1000`
- MacOS
  - `USER_UID=501`

#### .env

- devcontainer
  - .devcontainer/.env
- docker-compose
  - .env

### 실행

```bash
$ docker-compose up -d
```

### 개발

vscode의 devcontainer를 이용하며, 각 앱의 개발환경은 다음을 실행

- clsapp
  - Command Palette (F1)에서 **Remote-Containers: Open Folder in Container...** 실행
  - image-classification-with-transfer-learning/clsapp 열기
  - container 안에서 `$ go run main.go` 실행
- learnapp
  - Command Palette (F1)에서 **Remote-Containers: Open Folder in Container...** 실행
  - image-classification-with-transfer-learning/learnapp 열기
  - container 안에서 `$ python app.py` 실행

## APIs

### 모델

#### 모델 목록

`GET /models`

```bash
$ curl -XGET http://127.0.0.1:18080/models
```

최초 실행시 기본 모델 `default`를 생성

#### 모델 정보

`GET /models/:model`

```bash
$ curl -XGET http://127.0.0.1:18080/models/mymodel?verbose
```

- verbose (querystring)
  - 모델 정보 상세

#### 모델 생성

`POST /models/:model`

- subject (querystring)
  - 전이학습 이미지 그룹
- trial (querystring)
  - 전이학습 예제 모델
- epochs (querystring)
  - 학습 반복 횟수
- desc (querystring)
  - 모델 설명

기본 모델 생성

```bash
$ curl -XPOST http://127.0.0.1:18080/models/mymodel?desc=description
```

전이학습 시험 모델 생성

```bash
$ curl -XPOST http://127.0.0.1:18080/models/mymodel?trial&epochs=10
```

전이학습 모델 생성

```bash
$ curl -XPOST http://127.0.0.1:18080/models/mymodel?subject=flowers&epochs=10
```

#### 모델 삭제

`DELETE /models/:model`

```bash
$ curl -XDELETE http://127.0.0.1:18080/models/mymodel
```

### 이미지

#### 이미지 목록

`GET /images`

- subject (querystring)
  - 전이학습 이미지 그룹
- category (querystring)
  - 이미지 카테고리

```bash
$ curl -XGET http://127.0.0.1:18080/images?subject=flowers&category=roses
```

#### 이미지 추가

`POST /images`

- subject (querystring)
  - 전이학습 이미지 그룹
- category (querystring)
  - 이미지 카테고리
- images (multipart form)
  - 이미지 파일

```bash
$ curl -XPOST http://127.0.0.1:18080/images?subject=flowers&category=roses \
    -F 'images[]=@roses1.jpg' \
    -F 'images[]=@roses2.jpg'
```

#### 이미지 삭제

`DELETE /images`

- subject (querystring)
  - 전이학습 이미지 그룹
- category (querystring)
  - 이미지 카테고리

```bash
$ curl -XDELETE http://127.0.0.1:18080/images?subject=flowers&category=roses
```

### 추론

`POST /inference/:model`

- k (querystring)
  - 다중 카테고리 분류 모델에서 상위 카테고리 수
- image (multipart form)
  - 이미지 파일

```bash
$ curl -XPOST localhost:18080/inference/mymodel?k=10 \
    -F 'image=@roses.jpg'
```
