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
	user     = "postgres"
	password = "your-password"
	dbname   = "your-dbname"
)

func main() {
	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	err = db.Ping()
	if err != nil {
		log.Fatalf("Failed to ping database: %v", err)
	}

	log.Println("Successfully connected!")

	createTable(db)
	insertSampleData(db)
	retrieveLogs(db)
}

func createTable(db *sql.DB) {
	query := `
	CREATE TABLE IF NOT EXISTS logs (
		id SERIAL PRIMARY KEY,
		timestamp TIMESTAMP NOT NULL,
		message TEXT NOT NULL
	)`
	_, err := db.Exec(query)
	if err != nil {
		log.Fatalf("Failed to create table: %v", err)
	}
	log.Println("Table created successfully.")
}

func insertSampleData(db *sql.DB) {
	query := `INSERT INTO logs (timestamp, message) VALUES ($1, $2)`
	_, err := db.Exec(query, time.Now(), "This is a test log entry")
	if err != nil {
		log.Fatalf("Failed to insert sample data: %v", err)
	}
	log.Println("Sample data inserted successfully.")
}

func retrieveLogs(db *sql.DB) {
	query := `SELECT id, timestamp, message FROM logs`
	rows, err := db.Query(query)
	if err != nil {
		log.Fatalf("Failed to retrieve logs: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var timestamp time.Time
		var message string
		err := rows.Scan(&id, &timestamp, &message)
		if err != nil {
			log.Fatalf("Failed to read log entries: %v", err)
		}
		log.Printf("Log %d: %v - %s\n", id, timestamp, message)
	}
	err = rows.Err()
	if err != nil {
		log.Fatal(err)
	}
	log.Println("Logs retrieved successfully.")
}
