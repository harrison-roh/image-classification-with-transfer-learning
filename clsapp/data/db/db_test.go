package db

import (
	"database/sql"
	"fmt"
	"log"
	"testing"
)

type Result struct {
	no   int
	name string
}

func TestDB(t *testing.T) {
	driverName := "mysql"
	connInfo := "user1:password1@tcp(db:3306)/cls_image_db"
	tableName := "test_tab1"

	conn, err := New(Config{
		DriverName: driverName,
		ConnInfo:   connInfo,
		TableName:  tableName,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Destroy()
	log.Print(fmt.Sprintf("Init %s table", tableName))

	db, err := sql.Open(driverName, connInfo)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	log.Print("db status=", db.Stats())

	res, _ := db.Query("SHOW TABLES;")

	var table string

	for res.Next() {
		res.Scan(&table)
		log.Print("table=", table)
	}

	if _, err := db.Exec(fmt.Sprintf("DROP TABLE %s;", tableName)); err != nil {
		log.Fatal(err)
	}
}
