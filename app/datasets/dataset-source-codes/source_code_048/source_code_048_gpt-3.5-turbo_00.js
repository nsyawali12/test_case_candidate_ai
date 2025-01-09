function parseUrl(url) {
    const result = {
        queryParams: {}
    };

    // Get protocol
    const protocolIndex = url.indexOf('://');
    result.protocol = url.substring(0, protocolIndex);

    // Get host and port
    const afterProtocol = url.substring(protocolIndex + 3);
    const hostAndPort = afterProtocol.split('/')[0];
    const parts = hostAndPort.split(':');
    result.host = parts[0];
    result.port = parts[1] || '';

    // Get path
    const pathAndQuery = afterProtocol.substring(hostAndPort.length);
    const pathIndex = pathAndQuery.indexOf('/');
    result.path = pathAndQuery.substring(pathIndex);

    // Parse query parameters
    const queryIndex = pathAndQuery.indexOf('?');
    if (queryIndex !== -1) {
        const queryParams = pathAndQuery.substring(queryIndex + 1, pathIndex);
        queryParams.split('&').forEach(param => {
            const pair = param.split('=');
            result.queryParams[pair[0]] = pair[1];
        });
    }

    // Get anchor
    const anchorIndex = pathAndQuery.indexOf('#');
    if (anchorIndex !== -1) {
        result.anchor = pathAndQuery.substring(anchorIndex + 1);
    }

    return result;
}

// Example
const url = 'https://www.example.com:8080/path/to/page?name=ferret&color=purple#section2';
console.log(parseUrl(url));
