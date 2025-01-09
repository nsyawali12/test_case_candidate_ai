const https = require('https');

function getStatusCodes(urls) {
    return new Promise((resolve, reject) => {
        let result = {};

        // Using Promise.all to make concurrent HTTP requests
        Promise.all(urls.map((url) => {
            return new Promise((resolve, reject) => {
                https.get(url, (res) => {
                    // Storing the HTTP status code for each URL in the result object
                    result[url] = res.statusCode;
                    resolve();
                }).on('error', (err) => {
                    reject(err);
                });
            });
        })).then(() => {
            resolve(result);
        }).catch((err) => {
            reject(err);
        });
    });
}
