function parseURL(url) {
    const result = {};

    // retrieving the protocol
    const protocolSplit = url.split('://');
    result.protocol = protocolSplit[0];

    // everything after removing the protocol and '://'
    const remainingURL = protocolSplit[1];

    const sections = remainingURL.split('/');

    // working with the host and optional port
    const hostAndPort = sections.shift();
    const hostAndPortSplit = hostAndPort.split(':');
    result.host = hostAndPortSplit[0];
    if (hostAndPortSplit[1]) {
        result.port = hostAndPortSplit[1];
    }

    // the remaining part may contain path, query parameters and anchor (hash)
    const potentialPath = sections.join('/');

    // spliting path and the section that may contain query parameters and anchor
    const pathQueryParametersAnchor = potentialPath.split('?');
    result.path = '/' + pathQueryParametersAnchor[0];

    if (pathQueryParametersAnchor[1]) {
        // working with query parameters and anchor (hash)
        const queryParametersAnchor = pathQueryParametersAnchor[1].split('#');
        const queryParameters = queryParametersAnchor[0];

        const parameters = queryParameters.split('&');
        result.queryParams = parameters.reduce((acc, current) => {
            let keyAndValue = current.split('=');
            acc[keyAndValue[0]] = keyAndValue[1];
            return acc;
        }, {});

        if (queryParametersAnchor[1]) {
            // if there is an anchor (hash)
            result.anchor = queryParametersAnchor[1];
        }
    }

    return result;
}
