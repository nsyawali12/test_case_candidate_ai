function stripHtmlTags(str) {
  return str.replace(/<[^>]*>/g, '');
}

const input = '<p>Hello <em>World</em>!</p>';

const output = stripHtmlTags(input);

console.log(output);