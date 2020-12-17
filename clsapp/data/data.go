package data

import (
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"os"
	"path"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/harrison-roh/image-classification-with-transfer-learning/clsapp/constants"
	"github.com/harrison-roh/image-classification-with-transfer-learning/clsapp/data/db"
)

const (
	tableName  string = "image_tab"
	driverName string = "mysql"
	connInfo   string = "user1:password1@tcp(db:3306)/cls_image_db?parseTime=true"
)

// Manager 이미지 데이터를 관리
type Manager struct {
	Conn *db.DBconn
}

type saveFunc func(*multipart.FileHeader, string) error

func saveImage(file *multipart.FileHeader, dst string) error {
	src, err := file.Open()
	if err != nil {
		return err
	}
	defer src.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, src)

	return nil
}

// SaveImages image 저장
func (dm *Manager) SaveImages(subject, category string, images []*multipart.FileHeader, f saveFunc, verbose bool) (interface{}, error) {
	fileDir := path.Join(constants.ImagesPath, subject, category)
	if err := os.MkdirAll(fileDir, os.ModePerm); err != nil {
		return nil, err
	}

	if f == nil {
		f = saveImage
	}

	var (
		total      int64
		successful int64
		failed     int64
		items      []db.Item
		errors     []map[string]interface{}
	)
	for _, image := range images {
		total++

		orgFileName := image.Filename
		fileName := fmt.Sprintf("%s-%s", uuid.New().String()[:8], orgFileName)
		fileFormat := strings.ToLower(strings.Split(orgFileName, ".")[1])
		filePath := path.Join(fileDir, fileName)

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
			if verbose {
				errors = append(errors, map[string]interface{}{
					"orgfilename": orgFileName,
					"filename":    fileName,
					"error":       err.Error(),
				})
			}

			failed++
			continue
		}

		if err := f(image, filePath); err != nil {
			if verbose {
				errors = append(errors, map[string]interface{}{
					"orgfilename": orgFileName,
					"filename":    fileName,
					"error":       err.Error(),
				})
			}

			if _, err := dm.Conn.Delete(item); err != nil {
				log.Print(err)
			}

			failed++
			continue
		}

		if verbose {
			items = append(items, item)
		}
		successful++
	}

	infos := map[string]int64{
		"total":      total,
		"successful": successful,
		"failed":     failed,
	}

	result := make(map[string]interface{})
	result["infos"] = infos

	if verbose {
		result["images"] = items
	}

	if verbose {
		result["errors"] = errors
	}

	return result, nil
}

// DeleteImages image 삭제
func (dm *Manager) DeleteImages(subject, category, fileName, orgFileName string, verbose bool) (interface{}, error) {
	param := db.Item{
		Subject:     subject,
		Category:    category,
		Filename:    fileName,
		OrgFilename: orgFileName,
	}

	getInfos, items, err := dm.Conn.Get(param)
	if err != nil {
		return nil, err
	}

	getInfosMap := getInfos.(map[string]int64)
	if getInfosMap["total"] != getInfosMap["successful"] {
		return nil, fmt.Errorf(
			"Fail to read images %d of %d",
			getInfosMap["failed"],
			getInfosMap["total"],
		)
	}

	errors := make([]map[string]interface{}, 0)
	// 빈 디렉토리를 삭제하기 위해, subject와 category 목록을 저장
	scMap := make(map[string]map[string]int)
	for _, item := range items.([]db.Item) {
		if err := os.Remove(item.FilePath); err != nil {
			if verbose {
				errors = append(errors, map[string]interface{}{
					"orgfilename": item.OrgFilename,
					"filename":    item.Filename,
					"error":       err.Error(),
				})
			}
		} else {
			if _, ok := scMap[item.Subject]; !ok {
				scMap[item.Subject] = make(map[string]int)
			}
			if _, ok := scMap[item.Subject][item.Category]; !ok {
				scMap[item.Subject][item.Category] = 1
			} else {
				scMap[item.Subject][item.Category]++
			}
		}
	}

	deleted, err := dm.Conn.Delete(param)
	if err != nil {
		return nil, err
	}

	for subject := range scMap {
		for category := range scMap[subject] {
			categoryDir := path.Join(constants.ImagesPath, subject, category)
			// "directory not empty" 에러는 무시
			os.Remove(categoryDir)
		}

		subjectDir := path.Join(constants.ImagesPath, subject)
		// "directory not empty" 에러는 무시
		os.Remove(subjectDir)
	}

	infos := map[string]interface{}{
		"total":      getInfosMap["total"],
		"successful": deleted,
		"failed":     getInfosMap["total"] - deleted,
	}

	result := make(map[string]interface{})
	result["infos"] = infos

	if verbose {
		result["images"] = items
	}

	if verbose {
		result["errors"] = errors
	}

	return result, nil
}

// ListImages image 목록 반환
func (dm *Manager) ListImages(subject, category string) (interface{}, error) {
	param := db.Item{
		Subject:  subject,
		Category: category,
	}

	infos, items, err := dm.Conn.Get(param)
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
		log.Printf("DB %s close failed: %s", dm.Conn.TableName, err)
	} else {
		log.Printf("DB %s successfully closed", dm.Conn.TableName)
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
