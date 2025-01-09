function parseURL(url) {
    const result = {
        protocol: '',
        host: '',
        port: '',
        path: '',
        queryParams: {},
        anchor: ''
    };

    // Extract protocol
    const protocolEnd = url.indexOf("://");
    if (protocolEnd > -1) {
        result.protocol = url.substring(0, protocolEnd);
        url = url.substring(protocolEnd + 3);
    }

    // Extract anchor (hash)
    const hashStart = url.indexOf("#");
    if (hashStart > -1) {
        result.anchor = url.substring(hashStart + 1);
        url = url.substring(0, hashStart);
    }

    // Extract query parameters
    const queryStart = url.indexOf("?");
    if (queryStart > -1) {
        const queryString = url.substring(queryStart + 1);
        queryString.split("&").forEach(param => {
            const kv = param.split("=");
            result.queryParams[kv[0]] = kv[1] || '';
        });
        url = url.substring(0, queryStart);
    }

    // Extract port and host
    const portStart = url.lastIndexOf(":");
    const pathStart = url.indexOf("/");
    if (portStart > -1 && (portStart < pathStart || pathStart < 0)) {
        result.port = url.substring(portStart + 1, pathStart > -1 ? pathStart : undefined);
        result.host = url.substring(0, portStart);
    } else {
        result.host = url.substring(0, pathStart > -1 ? pathStart : undefined);
    }

    // Extract path
    if (pathStart > -1) {
        result.path = url.substring(pathStart);
    }

    return result;
}
