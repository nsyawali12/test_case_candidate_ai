const https = require('https');

// Define a function to fetch status codes for an array of URLs
function fetchStatusCodes(urls) {
    // Wrap all operations in a Promise
    return new Promise(resolve => {
        const results = {}; // Store statuses with URL as key
        let completedRequests = 0; // Track completed requests

        // Loop through each URL to initiate concurrent requests
        urls.forEach(url => {
            https.get(url, res => {
                results[url] = res.statusCode; // Store status code
                res.on('data', () => {}); // Must consume response data (even if not used)
                
                // On end of response data, check if all requests are done
                res.on('end', () => {
                    completedRequests++;
                    if (completedRequests === urls.length) {
                        resolve(results); // Resolve when all requests have finished
                    }
                });
            }).on('error', err => {
                // In case of error, store a default error status or customize as needed
                results[url] = 'Error'; 
                completedRequests++;
                if (completedRequests === urls.length) {
                    resolve(results); // Resolve even if there were errors
                }
            });
        });
    });
}

// Example Usage:
// fetchStatusCodes(['https://example.com', 'https://google.com'])
//     .then(result => console.log(result))
//     .catch(error => console.error(error));
