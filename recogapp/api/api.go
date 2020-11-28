package api

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/constants"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/data"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/inference"
)

// APIs api 핸들러
type APIs struct {
	I *inference.Inference
	M *data.Manager
}

// ListModels 추론 모델 목록 반환
func (a *APIs) ListModels(c *gin.Context) {
	models := a.I.GetModels()
	c.JSON(http.StatusOK, gin.H{
		"models": models,
	})
}

// ShowModel 추론 모델 정보 반환
func (a *APIs) ShowModel(c *gin.Context) {
	model := c.Param("model")

	if info := a.I.GetModel(model); info != nil {
		c.JSON(http.StatusOK, info)
	} else {
		Error(c, http.StatusBadRequest, fmt.Errorf("Cannot find model info: %s", model))
	}
}

// InferDefault 기본 모델을 이용한 추론
func (a *APIs) InferDefault(c *gin.Context) {
	a.infer(c, constants.DefaultModelName)
}

// InferWithModel 모델을 이용한 추론
func (a *APIs) InferWithModel(c *gin.Context) {
	model := c.Param("model")
	a.infer(c, model)
}

func (a *APIs) infer(c *gin.Context, model string) {
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		Error(c, http.StatusBadRequest, err)
		return
	}
	defer file.Close()

	var (
		image bytes.Buffer
		bytes int64
		n     int64
	)

	if n, err = io.Copy(&image, file); err != nil {
		Error(c, http.StatusBadRequest, err)
		return
	}
	bytes = n

	format := strings.Split(header.Filename, ".")[1]

	t0 := time.Now()
	if infers, err := a.I.Infer(model, image.String(), format, 5); err == nil {
		elapsed := time.Since(t0)
		c.JSON(http.StatusOK, gin.H{
			"file":        header.Filename,
			"format":      format,
			"bytes":       bytes,
			"inference":   infers,
			"elapsed(ms)": elapsed.Milliseconds(),
		})
	} else {
		Error(c, http.StatusBadRequest, err)
	}
}

// CreateModel model 생성
func (a *APIs) CreateModel(c *gin.Context) {
	model := c.Param("model")
	if model == "" {
		Error(c, http.StatusBadRequest, errors.New("Empty model name"))
		return
	}

	subject := c.PostForm("subject")
	desc := c.PostForm("desc")
	trial := c.PostForm("trial")
	isTrial, err := strconv.ParseBool(trial)
	if err != nil {
		isTrial = false
	}
	epochs := c.PostForm("epochs")
	nrEpochs, err := strconv.Atoi(epochs)
	if err != nil {
		nrEpochs = constants.TrainEpochs
	}

	if res, err := a.I.CreateModel(model, subject, desc, nrEpochs, isTrial); err != nil {
		Error(c, http.StatusInternalServerError, err)
	} else {
		c.JSON(http.StatusOK, res)
	}
}

// DeleteModel model 생성
func (a *APIs) DeleteModel(c *gin.Context) {
	model := c.Param("model")
	if model == "" {
		Error(c, http.StatusBadRequest, errors.New("Empty model name"))
		return
	}

	if err := a.I.DeleteModel(model); err != nil {
		Error(c, http.StatusInternalServerError, err)
	} else {
		c.String(http.StatusOK, "OK")
	}
}

// UploadImages image 업로드
func (a *APIs) UploadImages(c *gin.Context) {
	var (
		subject  string
		category string
	)
	if subject = c.PostForm("subject"); subject == "" {
		Error(c, http.StatusBadRequest, errors.New("Empty `subject`"))
		return
	}
	if category = c.PostForm("category"); category == "" {
		Error(c, http.StatusBadRequest, errors.New("Empty `category`"))
		return
	}

	form, err := c.MultipartForm()
	if err != nil {
		Error(c, http.StatusBadRequest, err)
		return
	}
	images := form.File["images[]"]

	if result, err := a.M.SaveImages(subject, category, images, c.SaveUploadedFile); err != nil {
		Error(c, http.StatusBadRequest, err)
	} else {
		c.JSON(http.StatusOK, result)
	}
}

// ListImages image 반환
func (a *APIs) ListImages(c *gin.Context) {
	subject := c.Query("subject")
	category := c.Query("category")

	if result, err := a.M.ListImages(subject, category); err != nil {
		Error(c, http.StatusBadRequest, err)
	} else {
		c.JSON(http.StatusOK, result)
	}
}

// HTTPError api 에러 메시지
type HTTPError struct {
	Error string `json:"error"`
}

// Error api 에러를 담은 json 응답 생성
func Error(c *gin.Context, status int, err error) {
	c.JSON(status, HTTPError{
		Error: err.Error(),
	})
}
