function stripHtmlTags(inputString) {
    // Use RegExp to remove tags and return only text content
    return inputString.replace(/<\/?[^>]+(>|$)/g, "");
}
