package api

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
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

// ListInference 추론 모델 목록 반환
func (a *APIs) ListInference(c *gin.Context) {
	models := a.I.GetModels()
	c.JSON(http.StatusOK, gin.H{
		"models": models,
	})
}

// ShowInference 추론 모델 정보 반환
func (a *APIs) ShowInference(c *gin.Context) {
	model := c.Param("model")

	if info := a.I.GetModel(model); info != nil {
		c.JSON(http.StatusOK, info)
	} else {
		Error(c, http.StatusBadRequest, fmt.Errorf("Cannot find model info: %s", model))
	}
}

// InferDefault 기본 모델을 이용한 추론
func (a *APIs) InferDefault(c *gin.Context) {
	a.doInfer(c, "default")
}

// InferWithModel 모델을 이용한 추론
func (a *APIs) InferWithModel(c *gin.Context) {
	model := c.Param("model")
	a.doInfer(c, model)
}

func (a *APIs) doInfer(c *gin.Context, model string) {
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
			"image":      header.Filename,
			"format":     format,
			"bytes":      bytes,
			"label":      infers[0].Label,
			"probabilty": infers[0].Prob,
		})
	} else {
		Error(c, http.StatusBadRequest, err)
	}
}

// UploadImage image를 업로드
func (a *APIs) UploadImage(c *gin.Context) {
	if item, err := a.M.Save(c); err != nil {
		Error(c, http.StatusBadRequest, err)
	} else {
		c.JSON(http.StatusOK, item)
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
