function stripHtmlTags(inputString) {
    // Using a regular expression to remove any HTML tags
    return inputString.replace(/<[^>]*>/g, '');
}
