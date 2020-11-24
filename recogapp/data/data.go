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
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/constants"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/data/db"
)

const (
	tableName  string = "image_tab"
	driverName string = "mysql"
	connInfo   string = "user1:password1@tcp(db:3306)/recog_image_db?parseTime=true"
)

// Manager 이미지 데이터를 관리
type Manager struct {
	Conn *db.DBconn
}

// SaveImages image 저장
func (dm *Manager) SaveImages(c *gin.Context) (interface{}, error) {
	var (
		subject  string
		category string
	)

	result := make(map[string]interface{})

	form, err := c.MultipartForm()
	if err != nil {
		return result, err
	}

	if subject = c.PostForm("subject"); subject == "" {
		return result, errors.New("Empty \"subject\"")
	}
	result["subject"] = subject

	if category = c.PostForm("category"); category == "" {
		return result, errors.New("Empty \"category\"")
	}
	result["category"] = category

	filePath := path.Join(constants.ImagesPath, subject, category)
	if err := os.MkdirAll(filePath, os.ModePerm); err != nil {
		return result, err
	}

	total := 0
	nrSuccessful := 0
	nrFailed := 0
	errors := make([]map[string]interface{}, 0)
	for _, image := range form.File["images[]"] {
		total++

		orgFileName := image.Filename
		fileName := fmt.Sprintf("%s-%s", uuid.New().String()[:8], orgFileName)
		fileFormat := strings.ToLower(strings.Split(orgFileName, ".")[1])

		if err := c.SaveUploadedFile(image, path.Join(filePath, fileName)); err != nil {
			errors = append(errors, map[string]interface{}{
				"file":  orgFileName,
				"error": err.Error(),
			})
			nrFailed++
			continue
		}

		item := db.Item{
			Subject:     subject,
			Category:    category,
			OrgFilename: orgFileName,
			Filename:    fileName,
			FileFormat:  fileFormat,
			FilePath:    filePath,
			CreateAt:    time.Now(),
		}

		if err := dm.Conn.Insert(item); err != nil {
			if err := os.Remove(path.Join(filePath, fileName)); err != nil {
				log.Print(err)
			}
			errors = append(errors, map[string]interface{}{
				"file":  orgFileName,
				"error": err.Error(),
			})
			nrFailed++
		} else {
			nrSuccessful++
		}
	}

	result["total"] = total
	result["failed"] = nrFailed
	result["successful"] = nrSuccessful
	result["errors"] = errors

	return result, nil
}

// ListImages image 목록 반환
func (dm *Manager) ListImages(subject, category string) (interface{}, error) {
	infos, items, err := dm.Conn.Get(subject, category)
	if err != nil {
		return nil, err
	}

	result := map[string]interface{}{
		"infos":  infos,
		"images": items,
	}

	return result, nil
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
