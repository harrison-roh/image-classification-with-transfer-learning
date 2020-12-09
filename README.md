## 전이학습을 이용한 이미지 분류

keras를 이용해 전이학습을 통한 모델을 생성하며, GO tensorflow를 이용해 모델 로드 및 이미지를 추론.
전이학습을 위한 사용자 이미지를 업로드 하고, 기본 모델 기반의 전이학습을 통해 분류계층을 학습함으로써 해당 이미지에 특화된 분류 모델을 생성.

## 시작하기

### 실행

```sh
docker-compose up -d
```

### 개발

vscode의 devcontainer를 이용하며, 각 앱의 개발환경은 다음을 실행

- clsapp
  - Command Palette (F1)에서 **Remote-Containers: Open Folder in Container...** 실행
  - image-classification-with-transfer-learning/clsapp 열기
  - container 안에서 `go run main.go` 실행
- learnapp
  - Command Palette (F1)에서 **Remote-Containers: Open Folder in Container...** 실행
  - image-classification-with-transfer-learning/learnapp 열기
  - container 안에서 `python app.py` 실행

## APIs

### 모델

#### 모델 목록

`GET /models`

```sh
curl -XGET http://127.0.0.1:18080/models
```

#### 모델 정보

`GET /models/:model`

```sh
curl -XGET http://127.0.0.1:18080/models/mymodel
```

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

```sh
curl -XPOST http://127.0.0.1:18080/models/mymodel?desc=description
```

전이학습 시험 모델 생성

```sh
curl -XPOST http://127.0.0.1:18080/models/mymodel?trial&epochs=10
```

전이학습 모델 생성

```sh
curl -XPOST http://127.0.0.1:18080/models/mymodel?subject=flowers&epochs=10
```

#### 모델 삭제

`DELETE /models/:model`

```sh
curl -XDELETE http://127.0.0.1:18080/models/mymodel
```

### 이미지

#### 이미지 목록

`GET /images`

- subject (querystring)
  - 전이학습 이미지 그룹
- category (querystring)
  - 이미지 카테고리 그룹

```sh
curl -XGET http://127.0.0.1:18080/images?subject=flowers&category=roses
```

#### 이미지 추가

`POST /images`

- subject (querystring)
  - 전이학습 이미지 그룹
- category (querystring)
  - 이미지 카테고리 그룹
- images (multipart form)
  - 이미지 파일

```sh
curl -XPOST http://127.0.0.1:18080/images?subject=flowers&category=roses \
    -F 'images[]=@roses1.jpg' \
    -F 'images[]=@roses2.jpg'
```

#### 이미지 삭제

`DELETE /images`

- subject (querystring)
  - 전이학습 이미지 그룹
- category (querystring)
  - 이미지 카테고리 그룹

```sh
curl -XDELETE http://127.0.0.1:18080/images?subject=flowers&category=roses
```

### 추론

`POST /inference/:model`

- image (multipart form)
  - 이미지 파일

```sh
curl -XPOST localhost:18080/inference/mymodel \
    -F 'image=@roses.jpg'
```
