const http = require('http');

async function checkUrlsStatus(urls) {
    const statusObj = {};

    const getStatus = async (url) => {
        return new Promise((resolve) => {
            http.get(url, (res) => {
                resolve(res.statusCode);
            }).on('error', (error) => {
                resolve(500); // Internal Server Error
            });
        });
    };

    await Promise.all(urls.map(async (url) => {
        const status = await getStatus(url);
        statusObj[url] = status;
    }));

    return statusObj;
}

// Example
const urls = ['https://example.com', 'https://google.com'];
checkUrlsStatus(urls)
    .then((result) => {
        console.log(result);
    })
    .catch((error) => {
        console.error(error);
    });

module.exports = checkUrlsStatus;
