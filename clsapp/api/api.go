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
	"github.com/harrison-roh/image-classification-with-transfer-learning/clsapp/constants"
	"github.com/harrison-roh/image-classification-with-transfer-learning/clsapp/data"
	"github.com/harrison-roh/image-classification-with-transfer-learning/clsapp/inference"
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

	subject := c.Query("subject")
	desc := c.Query("desc")
	_, trial := c.GetQuery("trial")
	epochs := c.Query("epochs")
	nrEpochs, err := strconv.Atoi(epochs)
	if err != nil {
		nrEpochs = constants.TrainEpochs
	}

	if res, err := a.I.CreateModel(model, subject, desc, nrEpochs, trial); err != nil {
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
	if subject = c.Query("subject"); subject == "" {
		Error(c, http.StatusBadRequest, errors.New("Empty `subject`"))
		return
	}
	if category = c.Query("category"); category == "" {
		Error(c, http.StatusBadRequest, errors.New("Empty `category`"))
		return
	}

	form, err := c.MultipartForm()
	if err != nil {
		Error(c, http.StatusBadRequest, err)
		return
	}
	images := form.File["images[]"]
	_, verbose := c.GetQuery("verbose")

	if result, err := a.M.SaveImages(subject, category, images, c.SaveUploadedFile, verbose); err != nil {
		Error(c, http.StatusBadRequest, err)
	} else {
		c.JSON(http.StatusOK, result)
	}
}

// DeleteImages image 삭제
func (a *APIs) DeleteImages(c *gin.Context) {
	subject := c.Query("subject")
	category := c.Query("category")
	fileName := c.Query("filename")
	orgFileName := c.Query("orgfilename")
	_, verbose := c.GetQuery("verbose")

	if result, err := a.M.DeleteImages(subject, category, fileName, orgFileName, verbose); err != nil {
		Error(c, http.StatusInternalServerError, err)
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
