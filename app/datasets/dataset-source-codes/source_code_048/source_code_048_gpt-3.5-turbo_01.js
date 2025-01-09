function parseURL(url) {
    let parsedURL = {};

    // Parse protocol
    parsedURL.protocol = url.split('://')[0];

    // Parse host and port
    let hostAndPort = url.split('/')[2];
    if (hostAndPort.includes(':')) {
        let hostPortArr = hostAndPort.split(':');
        parsedURL.host = hostPortArr[0];
        parsedURL.port = hostPortArr[1];
    } else {
        parsedURL.host = hostAndPort;
    }

    // Parse path
    let pathAndQuery = url.split('?')[0].split(parsedURL.host)[1];
    parsedURL.path = pathAndQuery.includes('/') ? pathAndQuery : '/';

    // Parse query parameters
    parsedURL.queryParams = {};
    if (url.includes('?')) {
        let queryParamsStr = url.split('?')[1].split('#')[0];
        let queryParamsArr = queryParamsStr.split('&');
        queryParamsArr.forEach(param => {
            let paramArr = param.split('=');
            parsedURL.queryParams[paramArr[0]] = paramArr[1];
        });
    }

    // Parse anchor
    if (url.includes('#')) {
        parsedURL.anchor = url.split('#')[1];
    }

    return parsedURL;
}

// Example
console.log(parseURL('https://www.example.com:8080/path/to/page?name=ferret&color=purple#section2'));
