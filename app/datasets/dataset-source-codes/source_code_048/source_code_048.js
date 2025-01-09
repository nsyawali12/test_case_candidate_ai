function parseUrl(url) {
    const urlRegex = /^(.*):\/\/([^:/]+)(?::(\d+))?(\/[^?#]*)?(?:\?([^#]*))?(?:#(.*))?$/;
    const matches = url.match(urlRegex);

    if (!matches) {
        throw new Error('Invalid URL format');
    }

    const [, protocol, host, port, path, queryParams, anchor] = matches;
    const queryParamObj = {};
    if (queryParams) {
        const queryParamsArr = queryParams.split('&');
        queryParamsArr.forEach(param => {
            const [key, value] = param.split('=');
            queryParamObj[key] = value || null;
        });
    }

    // Return object 
    return {
        protocol: protocol || null,
        host: host || null,
        port: port ? parseInt(port, 10) : null,
        path: path || null,
        queryParams: queryParamObj,
        anchor: anchor || null
    };
}



//another solution is but I prefer the top one

function parseURL(url) {
    const result = {};
    const protocolIndex = url.indexOf('://');
    result.protocol = url.slice(0, protocolIndex);
    const urlWithoutProtocol = url.slice(protocolIndex + 3);

    //  anchor
    const anchorIndex = urlWithoutProtocol.indexOf('#');
    if (anchorIndex !== -1) {
        result.anchor = urlWithoutProtocol.slice(anchorIndex + 1);
    }

   
    const queryIndex = urlWithoutProtocol.indexOf('?');
    if (queryIndex !== -1) {
        const queryParamsString = urlWithoutProtocol.slice(queryIndex + 1, anchorIndex !== -1 ? anchorIndex : undefined);
        const queryParamsArray = queryParamsString.split('&');
        result.queryParams = {};
        queryParamsArray.forEach(param => {
            const [key, value] = param.split('=');
            result.queryParams[key] = value;
        });
    }

   
    const urlWithoutQueryParams = queryIndex !== -1 ? urlWithoutProtocol.slice(0, queryIndex) : urlWithoutProtocol;
    const [hostAndPort, path] = urlWithoutQueryParams.split('/');
    result.path = '/' + path;
    const [host, port] = hostAndPort.split(':');
    result.host = host;
    result.port = port || '';

    return result;
}


//usage
const url = 'https://www.example.com:8080/path/to/page?name=ferret&color=purple#section2';
console.log(parseUrl(url));