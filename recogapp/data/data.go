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
	driverName    string = "mysql"
	connInfo      string = "user1:password1@tcp(db:3306)/recog_image_db"
)

// Manager 이미지 데이터를 관리
type Manager struct {
	Conn *db.DBconn
}

// ImageItem 이미지 저장 정보
type ImageItem struct {
	Model      string    `json:"model"`
	Category   string    `json:"category"`
	Filename   string    `json:"filename"`
	FileFormat string    `json:"format"`
	FilePath   string    `json:"path"`
	CreateAt   time.Time `json:"createAt"`
}

// Save image 저장
func (dm *Manager) Save(c *gin.Context) (ImageItem, error) {
	var (
		model    string
		category string
		item     ImageItem
	)

	file, header, err := c.Request.FormFile("image")
	if err != nil {
		return item, err
	}
	defer file.Close()

	if model = c.PostForm("model"); model == "" {
		return item, errors.New("Empty \"model\"")
	}
	if category = c.PostForm("category"); category == "" {
		return item, errors.New("Empty \"category\"")
	}

	fileName := fmt.Sprintf("%s-%s", uuid.New().String()[:8], header.Filename)
	fileFormat := strings.ToLower(strings.Split(header.Filename, ".")[1])
	filePath := path.Join(rootImagePath, model, category)

	if err := os.MkdirAll(filePath, os.ModePerm); err != nil {
		return item, err
	}

	if err := c.SaveUploadedFile(header, path.Join(filePath, fileName)); err != nil {
		return item, err
	}

	item = ImageItem{
		Model:      model,
		Category:   category,
		Filename:   fileName,
		FileFormat: fileFormat,
		FilePath:   filePath,
		CreateAt:   time.Now(),
	}

	return item, dm.Conn.Insert(db.Item(item))
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
func New(tableName string) (*Manager, error) {
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
