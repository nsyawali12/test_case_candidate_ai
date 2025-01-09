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
	dbname   = "your_database"
)

func connectDB() (*sql.DB, error) {
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable", host, port, user, password, dbname)
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
	createTableSQL := `
		CREATE TABLE IF NOT EXISTS logs (
			id SERIAL PRIMARY KEY,
			timestamp TIMESTAMPTZ NOT NULL,
			message TEXT
		)
	`
	_, err := db.Exec(createTableSQL)
	return err
}

func insertSampleData(db *sql.DB) error {
	insertDataSQL := `
		INSERT INTO logs (timestamp, message) VALUES ($1, $2), ($3, $4)
	`
	_, err := db.Exec(insertDataSQL, time.Now(), "Sample log entry 1", time.Now().Add(1*time.Hour), "Sample log entry 2")
	return err
}

func retrieveLogs(db *sql.DB) error {
	query := `
		SELECT * FROM logs
	`
	rows, err := db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var timestamp time.Time
		var message string
		if err := rows.Scan(&id, &timestamp, &message); err != nil {
			return err
		}
		fmt.Printf("ID: %d, Timestamp: %v, Message: %s\n", id, timestamp, message)
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
		log.Fatalf("Error creating 'logs' table: %v", err)
	}

	err = insertSampleData(db)
	if err != nil {
		log.Fatalf("Error inserting sample data: %v", err)
	}

	err = retrieveLogs(db)
	if err != nil {
		log.Fatalf("Error retrieving logs: %v", err)
	}
}
