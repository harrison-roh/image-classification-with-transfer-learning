package db

import (
	"database/sql"
	"fmt"
	"log"
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

// Item 데이터 항목
type Item struct {
	Model       string
	Category    string
	OrgFilename string
	Filename    string
	FileFormat  string
	FilePath    string
	CreateAt    time.Time
}

func (conn *DBconn) createTable() error {
	if _, err := conn.db.Exec(fmt.Sprintf(`CREATE TABLE %s (
		model CHAR(20) NOT NULL,
		category CHAR(20) NOT NULL,
		orgfilename Char(20) NOT NULL,
		filename Char(20) NOT NULL,
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
		model,
		category,
		orgfilename,
		filename,
		format,
		path,
		createAt) value (?, ?, ?, ?, ?, ?, ?);`, conn.TableName),
		item.Model, item.Category, item.OrgFilename, item.Filename,
		item.FileFormat, item.FilePath, createAt,
	)

	return err
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
