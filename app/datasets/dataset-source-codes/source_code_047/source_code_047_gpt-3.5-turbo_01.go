package main

import (
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

const (
	host     = "localhost"
	port     = 5432
	user     = "your_username"
	password = "your_password"
	dbname   = "your_dbname"
)

func connectDB() (*sql.DB, error) {
	connStr := fmt.Sprintf("host=%s port=%d user=%s "+"password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)

	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, err
	}

	err = db.Ping()
	if err != nil {
		return nil, err
	}

	return db, nil
}

func createLogsTable(db *sql.DB) error {
	_, err := db.Exec(`CREATE TABLE IF NOT EXISTS logs (
		id SERIAL PRIMARY KEY,
		timestamp TIMESTAMP NOT NULL,
		message TEXT NOT NULL
	);`)
	return err
}

func insertLog(db *sql.DB, message string) error {
	_, err := db.Exec("INSERT INTO logs (timestamp, message) VALUES ($1, $2);",
		time.Now(), message)
	return err
}

func retrieveLogs(db *sql.DB) error {
	rows, err := db.Query("SELECT * FROM logs;")
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var timestamp time.Time
		var message string
		err = rows.Scan(&id, &timestamp, &message)
		if err != nil {
			return err
		}
		fmt.Printf("ID: %d, Timestamp: %s, Message: %s\n", id, timestamp, message)
	}

	return nil
}

func main() {
	db, err := connectDB()
	if err != nil {
		log.Fatalf("Error connecting to the database: %v", err)
	}
	defer db.Close()

	err = createLogsTable(db)
	if err != nil {
		log.Fatalf("Error creating logs table: %v", err)
	}

	err = insertLog(db, "Sample log entry 1")
	if err != nil {
		log.Fatalf("Error inserting log entry: %v", err)
	}

	err = insertLog(db, "Sample log entry 2")
	if err != nil {
		log.Fatalf("Error inserting log entry: %v", err)
	}

	err = retrieveLogs(db)
	if err != nil {
		log.Fatalf("Error retrieving logs: %v", err)
	}
}
