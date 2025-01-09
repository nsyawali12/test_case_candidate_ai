const https = require('https');

function fetchStatusCodes(urls) {
  // Create a function that returns a Promise
  return new Promise((resolve, reject) => {
    let results = {}; // Object to store URL and its corresponding status code
    let completedRequests = 0; // Counter to track completed requests

    urls.forEach((url) => {
      https.get(url, (res) => {
        // Increment the completed requests counter once each request finishes
        completedRequests++;
        
        // Store the status code of the URL
        results[url] = res.statusCode;

        // If all requests are completed, resolve the Promise
        if (completedRequests === urls.length) {
          resolve(results);
        }
      }).on('error', (err) => {
        // In case of an error, store 'null' as the status code for that URL
        completedRequests++;
        results[url] = null;

        // Still continue with other requests, unless all failed
        if (completedRequests === urls.length) {
          resolve(results);
        }
      });
    });
  });
}

// Example usage:
// fetchStatusCodes(['https://example.com', 'https://google.com']).then(console.log);
