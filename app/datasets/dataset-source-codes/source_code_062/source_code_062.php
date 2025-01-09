<?php
function getTopThreeWords($text) {
// Remove punctuation and convert to lowercase
$words = str_word_count(strtolower(preg_replace('/[^\p{L}\p{N}\s]/u', ' ', $text)), 1);

// Count the frequency of each word
$wordFrequency = array_count_values($words);

// Sort the words by frequency in descending order
arsort($wordFrequency);

// Get the top three words
$topThreeWords = array_slice($wordFrequency, 0, 3, true);

// Format the output
$output = [];
foreach ($topThreeWords as $word => $count) {
$output[] = "('$word', $count)";
}

return '[' . implode(', ', $output) . ']';
}

// Example usage:
$inputText = "The quick brown fox jumps over the lazy dog. The dog was lazy!";
echo getTopThreeWords($inputText);
?>