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
    port     = 5432 // Default PostgreSQL port
    user     = "your_username"
    password = "your_password"
    dbname   = "your_database"
)

func main() {
    // Connection string
    psqlInfo := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
        host, port, user, password, dbname)

    // Connect to the database
    db, err := sql.Open("postgres", psqlInfo)
    if err != nil {
        log.Fatalf("Failed to connect to database: %v", err)
    }
    defer db.Close()

    // Check the connection
    err = db.Ping()
    if err != nil {
        log.Fatalf("Failed to ping database: %v", err)
    }

    // Create logs table
    createTableSQL := `CREATE TABLE IF NOT EXISTS logs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        message TEXT NOT NULL
    );`
    _, err = db.Exec(createTableSQL)
    if err != nil {
        log.Fatalf("Failed to create table: %v", err)
    }

    // Insert data into the table
    insertSQL := `INSERT INTO logs (timestamp, message) VALUES ($1, $2)`
    _, err = db.Exec(insertSQL, time.Now(), "This is the first log message")
    if err != nil {
        log.Fatalf("Failed to insert data into logs table: %v", err)
    }

    // Query all entries from logs
    rows, err := db.Query("SELECT id, timestamp, message FROM logs")
    if err != nil {
        log.Fatalf("Failed to query data from logs table: %v", err)
    }
    defer rows.Close()

    // Print all log entries
    for rows.Next() {
        var id int
        var timestamp time.Time
        var message string
        err := rows.Scan(&id, &timestamp, &message)
        if err != nil {
            log.Fatal("Failed to read data from query result: %v", err)
        }
        fmt.Printf("ID: %d, Timestamp: %s, Message: %s\n", id, timestamp, message)
    }

    // Check for any error experienced during iteration
    err = rows.Err()
    if err != nil {
        log.Fatal("Encountered error during row iteration: %v", err)
    }
}
