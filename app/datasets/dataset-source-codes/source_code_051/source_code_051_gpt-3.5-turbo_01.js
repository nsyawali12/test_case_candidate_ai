const http = require('http');

function getStatusCodes(urls) {
    const requests = urls.map(url => new Promise((resolve) => {
        http.get(url, (res) => {
            resolve({ url, statusCode: res.statusCode });
        }).on('error', (error) => {
            resolve({ url, statusCode: 500 }); // Handle request errors
        });
    }));

    return Promise.all(requests).then((results) => {
        const statusCodes = {};

        results.forEach((result) => {
            statusCodes[result.url] = result.statusCode;
        });

        return statusCodes;
    });
}

// Example usage
const urls = ['https://example.com', 'https://google.com'];
getStatusCodes(urls).then((response) => {
    console.log(response); // Output: {'https://example.com': 200, 'https://google.com': 200}
}).catch((error) => {
    console.error(error);
});
