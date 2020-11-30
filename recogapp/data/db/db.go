package db

import (
	"database/sql"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

// Config DBconn config
type Config struct {
	DriverName string
	ConnInfo   string

	TableName string
}

// DBconn db 연결정보
type DBconn struct {
	DriverName string
	ConnInfo   string

	TableName string

	db *sql.DB
}

type NullTime struct {
	sql.NullTime
}

func (nt *NullTime) MarshalJSON() ([]byte, error) {
	if !nt.Valid {
		return []byte("null"), nil
	}
	val := fmt.Sprintf("\"%s\"", nt.Time.Format(time.RFC3339))
	return []byte(val), nil
}

// Item 데이터 항목
type Item struct {
	Subject     string    `json:"subject"`
	Category    string    `json:"category"`
	OrgFilename string    `json:"orgfilename"`
	Filename    string    `json:"filename"`
	FileFormat  string    `json:"-"`
	FilePath    string    `json:"-"`
	CreateAt    time.Time `json:"createAt"`
}

func (conn *DBconn) createTable() error {
	if _, err := conn.db.Exec(fmt.Sprintf(`CREATE TABLE %s (
		subject CHAR(20) NOT NULL,
		category CHAR(20) NOT NULL,
		orgfilename Char(60) NOT NULL,
		filename Char(60) NOT NULL,
		format Char(10) NOT NULL,
		path VARCHAR(80) NOT NULL,
		createAt DATETIME NOT NULL);`, conn.TableName)); err != nil {
		return err
	}

	return nil
}

func (conn *DBconn) existsTable() bool {
	if _, err := conn.db.Query(fmt.Sprintf("SELECT * FROM %s;", conn.TableName)); err != nil {
		return false
	}

	return true
}

func (conn *DBconn) initTable() error {
	if !conn.existsTable() {
		log.Printf("Create DB table: %s", conn.TableName)
		return conn.createTable()
	}

	return nil
}

// Insert entry 삽입
func (conn *DBconn) Insert(item Item) error {
	createAt := item.CreateAt.Format(time.RFC3339)

	_, err := conn.db.Exec(fmt.Sprintf(`INSERT INTO %s (
		subject,
		category,
		orgfilename,
		filename,
		format,
		path,
		createAt) value (?, ?, ?, ?, ?, ?, ?);`, conn.TableName),
		item.Subject, item.Category, item.OrgFilename, item.Filename,
		item.FileFormat, item.FilePath, createAt,
	)

	return err
}

// Delete entry 삭제
func (conn *DBconn) Delete(param Item) (int64, error) {
	var where []string

	where = appendWhere(where, param.Subject, "subject")
	where = appendWhere(where, param.Category, "category")
	where = appendWhere(where, param.OrgFilename, "orgfilename")
	where = appendWhere(where, param.Filename, "filename")
	if len(where) == 0 {
		return -1, errors.New("No arguments")
	}

	whereQuery := strings.Join(where, " AND ")

	query := fmt.Sprintf("DELETE FROM %s WHERE %s", conn.TableName, whereQuery)

	result, err := conn.db.Exec(query)
	if err != nil {
		return -1, err
	}
	rows, _ := result.RowsAffected()

	return rows, nil
}

// Get entry 반환
func (conn *DBconn) Get(param Item) (interface{}, interface{}, error) {
	var where []string

	where = appendWhere(where, param.Subject, "subject")
	where = appendWhere(where, param.Category, "category")
	where = appendWhere(where, param.OrgFilename, "orgfilename")
	where = appendWhere(where, param.Filename, "filename")

	columns := "subject,category,filename,orgfilename,path,createAt"

	var query string
	if len(where) == 0 {
		query = fmt.Sprintf("SELECT %s FROM %s", columns, conn.TableName)
	} else {
		whereQuery := strings.Join(where, " AND ")
		query = fmt.Sprintf("SELECT %s FROM %s WHERE %s", columns, conn.TableName, whereQuery)
	}

	rows, err := conn.db.Query(query)
	if err != nil {
		return nil, nil, err
	}

	var (
		total      int64
		successful int64
		failed     int64
	)

	var items []Item
	for rows.Next() {
		total++
		item := Item{}
		if err := rows.Scan(
			&item.Subject,
			&item.Category,
			&item.Filename,
			&item.OrgFilename,
			&item.FilePath,
			&item.CreateAt); err != nil {
			failed++
			log.Print(err)
			continue
		}

		successful++
		items = append(items, item)
	}

	infos := map[string]int64{
		"total":      total,
		"successful": successful,
		"failed":     failed,
	}

	return infos, items, nil
}

func appendWhere(l []string, val, col string) []string {
	if val != "" {
		return append(l, fmt.Sprintf("%s='%s'", col, val))
	}

	return l
}

// Destroy db connection 해제
func (conn *DBconn) Destroy() error {
	return conn.db.Close()
}

// New 새로운 db connection 생성
func New(cfg Config) (*DBconn, error) {
	db, err := sql.Open(cfg.DriverName, cfg.ConnInfo)
	if err != nil {
		return nil, err
	}

	conn := &DBconn{
		DriverName: cfg.DriverName,
		ConnInfo:   cfg.ConnInfo,
		TableName:  cfg.TableName,
		db:         db,
	}

	if err := conn.initTable(); err != nil {
		db.Close()
		return nil, err
	}

	return conn, nil
}
