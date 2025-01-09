package main

import (
	"database/sql"
	"fmt"
	"log"

	_ "github.com/lib/pq"
)

func main() {
	// Connect to the database
	db, err := sql.Open("postgres", "user=postgres dbname=mydatabase sslmode=disable")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Create the logs table if it doesn't exist
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS logs (
		id SERIAL PRIMARY KEY,
		message TEXT
	)`)
	if err != nil {
		log.Fatal(err)
	}

	// Insert sample data into the logs table
	_, err = db.Exec("INSERT INTO logs (message) VALUES ($1), ($2), ($3)", "Log entry 1", "Log entry 2", "Log entry 3")
	if err != nil {
		log.Fatal(err)
	}

	// Retrieve all log entries from the logs table
	rows, err := db.Query("SELECT * FROM logs")
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	// Iterate over the rows and print each log entry
	fmt.Println("Log Entries:")
	for rows.Next() {
		var id int
		var message string
		err := rows.Scan(&id, &message)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("ID: %d, Message: %s\n", id, message)
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}
}