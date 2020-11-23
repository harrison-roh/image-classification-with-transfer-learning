package api

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
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
	a.infer(c, "default")
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
	}
	defer file.Close()

	var (
		image bytes.Buffer
		bytes int64
	)

	if n, err := io.Copy(&image, file); err != nil {
		Error(c, http.StatusBadRequest, err)
	} else {
		bytes = n
	}

	format := strings.Split(header.Filename, ".")[1]

	if infers, err := a.I.Infer(model, image.String(), format, 5); err == nil {
		c.JSON(http.StatusOK, gin.H{
			"image":          header.Filename,
			"format":         format,
			"bytes":          bytes,
			"top label":      infers[0].Label,
			"top probabilty": infers[0].Prob,
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
	}

	subject := c.PostForm("subject")
	desc := c.PostForm("desc")
	trial := c.PostForm("trial")
	isTrial, err := strconv.ParseBool(trial)
	if err != nil {
		isTrial = false
	}

	if res, err := a.I.CreateModel(model, subject, desc, isTrial); err != nil {
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
	}

	if err := a.I.DeleteModel(model); err != nil {
		Error(c, http.StatusInternalServerError, err)
	} else {
		c.String(http.StatusOK, "OK")
	}
}

// UploadImages image 업로드
func (a *APIs) UploadImages(c *gin.Context) {
	if result, err := a.M.SaveImages(c); err != nil {
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
