package data

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/data/db"
)

const (
	rootImagePath string = "/recog/images"
	tableName     string = "image_tab"
	driverName    string = "mysql"
	connInfo      string = "user1:password1@tcp(db:3306)/recog_image_db"
)

// Manager 이미지 데이터를 관리
type Manager struct {
	Conn *db.DBconn
}

// ImageItem 이미지 저장 정보
type ImageItem struct {
	Model       string    `json:"model"`
	Category    string    `json:"category"`
	OrgFilename string    `json:"orgfilename"`
	Filename    string    `json:"filename"`
	FileFormat  string    `json:"format"`
	FilePath    string    `json:"path"`
	CreateAt    time.Time `json:"createAt"`
}

// Save image 저장
func (dm *Manager) Save(c *gin.Context) (interface{}, error) {
	var (
		model    string
		category string
		item     ImageItem
	)

	image, err := c.FormFile("image")
	if err != nil {
		return item, err
	}

	if model = c.PostForm("model"); model == "" {
		return item, errors.New("Empty \"model\"")
	}
	if category = c.PostForm("category"); category == "" {
		return item, errors.New("Empty \"category\"")
	}

	orgFileName := image.Filename
	fileName := fmt.Sprintf("%s-%s", uuid.New().String()[:8], orgFileName)
	fileFormat := strings.ToLower(strings.Split(orgFileName, ".")[1])
	filePath := path.Join(rootImagePath, model, category)

	if err := os.MkdirAll(filePath, os.ModePerm); err != nil {
		return item, err
	}

	if err := c.SaveUploadedFile(image, path.Join(filePath, fileName)); err != nil {
		return item, err
	}

	item = ImageItem{
		Model:       model,
		Category:    category,
		OrgFilename: orgFileName,
		Filename:    fileName,
		FileFormat:  fileFormat,
		FilePath:    filePath,
		CreateAt:    time.Now(),
	}

	return item, dm.Conn.Insert(db.Item(item))
}

// SaveMultiple 복수의 image 저장
func (dm *Manager) SaveMultiple(c *gin.Context) (interface{}, error) {
	var (
		model    string
		category string
		item     ImageItem
	)

	result := map[string]interface{}{
		"model":      "",
		"category":   "",
		"path":       "",
		"total":      0,
		"successful": 0,
		"failed":     0,
	}

	form, err := c.MultipartForm()
	if err != nil {
		return item, err
	}

	if model = c.PostForm("model"); model == "" {
		return result, errors.New("Empty \"model\"")
	}
	result["model"] = model

	if category = c.PostForm("category"); category == "" {
		return result, errors.New("Empty \"category\"")
	}
	result["category"] = category

	filePath := path.Join(rootImagePath, model, category)
	if err := os.MkdirAll(filePath, os.ModePerm); err != nil {
		return result, err
	}
	result["path"] = filePath

	total := 0
	nrSuccessful := 0
	nrFailed := 0
	for _, image := range form.File["images[]"] {
		total++

		orgFileName := image.Filename
		fileName := fmt.Sprintf("%s-%s", uuid.New().String()[:8], orgFileName)
		fileFormat := strings.ToLower(strings.Split(orgFileName, ".")[1])

		if err := c.SaveUploadedFile(image, path.Join(filePath, fileName)); err != nil {
			log.Print(err)
			nrFailed++
			continue
		}

		item = ImageItem{
			Model:       model,
			Category:    category,
			OrgFilename: orgFileName,
			Filename:    fileName,
			FileFormat:  fileFormat,
			FilePath:    filePath,
			CreateAt:    time.Now(),
		}

		if err := dm.Conn.Insert(db.Item(item)); err != nil {
			log.Print(err)
			nrFailed++
		} else {
			nrSuccessful++
		}
	}

	result["total"] = total
	result["failed"] = nrFailed
	result["successful"] = nrSuccessful

	return result, nil
}

func (dm *Manager) save() error {
	return nil
}

// Destroy Data manager 해제
func (dm *Manager) Destroy() {
	if err := dm.Conn.Destroy(); err != nil {
		log.Printf("DB %s successfully closed", dm.Conn.TableName)
	} else {
		log.Printf("DB %s close failed", dm.Conn.TableName)
	}
}

// New 새로운 Data manager 생성
func New() (*Manager, error) {
	conn, err := db.New(db.Config{
		DriverName: driverName,
		ConnInfo:   connInfo,
		TableName:  tableName,
	})
	if err != nil {
		return nil, err
	}
	log.Printf("DB %s successfully initialized", tableName)

	dm := &Manager{
		Conn: conn,
	}

	return dm, nil
}