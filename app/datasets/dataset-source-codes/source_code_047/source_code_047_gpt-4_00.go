package main

import (
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

func main() {
	db, err := sql.Open("postgres", "user=postgres password=your-password dbname=test sslmode=disable")

	// check connection error
	if err != nil {
		log.Fatal(err)
	}

	// create table
	_, err = db.Exec("CREATE TABLE logs (id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE NOT NULL, message TEXT NOT NULL)")

	// handle error
	if err != nil {
		log.Fatal(err)
	}

	// insert data
	_, err = db.Exec("INSERT INTO logs (timestamp, message) VALUES ($1, $2)", time.Now(), "Test log message")

	// handle error
	if err != nil {
		log.Fatal(err)
	}

	// select all log entries
	rows, err := db.Query("SELECT * FROM logs")

	// handle error
	if err != nil {
		log.Fatal(err)
	}

	defer rows.Close()
	for rows.Next() {
		var id int
		var timestamp time.Time
		var message string

		// handle error
		err := rows.Scan(&id, &timestamp, &message)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println(id, timestamp, message)
	}

	// check for errors encountered while iterating over rows
	err = rows.Err()
	if err != nil {
		log.Fatal(err)
	}
}
