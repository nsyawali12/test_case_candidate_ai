function parseUrl(url) {
    var result = {};
    var match = url.match(/^([^:]+):\/\/([^:\/]+)(?::(\d+))?([^?#]*)(?:\?([^#]*))?(?:#(.*))?/);
    
    if (!match) {
        return null;
    }
  
    // protocol
    result.protocol = match[1];
  
    // host
    result.host = match[2];
  
    // port
    result.port = match[3];
  
    // path
    result.path = match[4];
  
    // anchor
    result.anchor = match[6];

    // queryParams  
    var queryParams = {};
    if (match[5]) {
        match[5].split('&').forEach(function(part) {
            var item = part.split('=');
            queryParams[item[0]] = decodeURIComponent(item[1]);
        });
    }
    result.queryParams = queryParams; 

    return result;
}
