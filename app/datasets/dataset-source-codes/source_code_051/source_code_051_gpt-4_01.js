const http = require('http');
const https = require('https');

function getStatusCodes(urls) {
  // Initialize an empty object to store the URLs and their corresponding status codes
  const statusCodes = {};

  // Return a new promise
  return new Promise((resolve, reject) => {
    // Track how many requests have completed
    let completedRequests = 0;

    // For each URL, make a concurrent HTTP request
    for (let i = 0; i < urls.length; i++) {
      const url = urls[i];

      // Choose the correct module based on the URL protocol
      const module = url.startsWith('https') ? https : http;

      // Make a GET request to the URL 
      module.get(url, res => {
        // Once the response is received, store the status code
        statusCodes[url] = res.statusCode;

        // Increment the completed request count
        completedRequests++;

        // If all requests have completed, resolve the promise
        if (completedRequests === urls.length) {
          resolve(statusCodes);
        }
      }).on('error', err => {
        // If an error occurs, reject the promise
        reject(err);
      });
    }
  });
}
