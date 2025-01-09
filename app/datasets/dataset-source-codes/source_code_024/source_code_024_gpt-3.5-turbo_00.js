function stripHtmlTags(input) {
    if (!input) return '';

    const tagRegex = /<[^>]*>/g;
    return input.replace(tagRegex, '');
}
