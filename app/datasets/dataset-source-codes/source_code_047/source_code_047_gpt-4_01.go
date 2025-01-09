// Importing Necessary Packages
import (
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

// Function for Database Operations
func handleDatabaseOperations() {
	
	// Set up the database connection
	db, err := sql.Open("postgres", "user=username dbname=database password=password sslmode=disable")

	// Error handling for database connection
	if err != nil {
		log.Fatal("Failed to open a DB connection: ", err)
	}

	// Close database connection at the end
	defer db.Close()

	// Create a new table called 'logs'
	_, err = db.Exec("CREATE TABLE logs (id INT PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE NOT NULL, message TEXT NOT NULL)")

	// Error handling for table creation
	if err != nil {
		log.Fatal("Failed to create table: ", err)
	}

	// Insert sample data into 'logs'.
	currentTime := time.Now()
	_, err = db.Exec(`INSERT INTO logs (id, timestamp, message)
                      VALUES ($1, $2, $3)`, 1, currentTime, "Sample log entry")

	// Error handling for data insertion
	if err != nil {
		log.Fatal("Failed to insert data: ", err)
	}

	// Query to retrieve all entries from table 'logs'
	rows, err := db.Query("SELECT * FROM logs")

	// Error handling for data retrieval
	if err != nil {
		log.Fatal("Failed to execute query: ", err)
	}

	// Close rows after scanning
	defer rows.Close()

	// Iterate through the results
	for rows.Next() {
		var (
			id        int
			timestamp time.Time
			message   string
		)
		err = rows.Scan(&id, &timestamp, &message)

		// Error handling for scanning rows
		if err != nil {
			log.Fatal("Failed to scan row: ", err)
		}

		// Printing the database entries
		fmt.Println(id, timestamp, message)
	}

	// Error Handling if any error occurred while iterating through the results
	if err = rows.Err(); err != nil {
		log.Fatal("An error occurred: ", err)
	}
}
