function parseURL(url) {
    const result = {
        protocol: '',
        host: '',
        port: '',
        path: '',
        queryParams: {},
        anchor: ''
    };

    // Extract the protocol
    const protocolEndIndex = url.indexOf('://');
    if (protocolEndIndex !== -1) {
        result.protocol = url.substring(0, protocolEndIndex);
        url = url.substring(protocolEndIndex + 3);
    }

    // Extract the anchor (hash)
    const hashIndex = url.indexOf('#');
    if (hashIndex !== -1) {
        result.anchor = url.substring(hashIndex + 1);
        url = url.substring(0, hashIndex);
    }

    // Extract the query parameters
    const queryParamsIndex = url.indexOf('?');
    if (queryParamsIndex !== -1) {
        const queryParamsString = url.substring(queryParamsIndex + 1);
        url = url.substring(0, queryParamsIndex);
        const allParams = queryParamsString.split('&');
        allParams.forEach(param => {
            const [key, value] = param.split('=');
            result.queryParams[key] = value;
        });
    }

    // Extract port if it exists
    const portIndex = url.lastIndexOf(':');
    const pathIndex = url.indexOf('/');
    if (portIndex !== -1 && portIndex < pathIndex) {
        result.port = url.substring(portIndex + 1, pathIndex);
        url = url.substring(0, portIndex);
    }

    // Extract host and path
    const pathStartIndex = url.indexOf('/');
    if (pathStartIndex !== -1) {
        result.host = url.substring(0, pathStartIndex);
        result.path = url.substring(pathStartIndex);
    } else { // No path, URL ended with host and possibly port
        result.host = url;
    }

    return result;
}
