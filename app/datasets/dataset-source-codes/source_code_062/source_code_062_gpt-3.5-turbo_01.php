<?php

function findTopThreeWords($input) {
    $input = strtolower($input); // Convert input string to lowercase
    $words = str_word_count($input, 1); // Create an array of words from the input
    
    // Remove punctuation from words
    $words = array_map(function($word) {
        return preg_replace("/[^A-Za-z]/", '', $word);
    }, $words);

    // Count the frequency of each word
    $wordCounts = array_count_values($words);
    
    // Sort the word counts in descending order
    arsort($wordCounts);
    
    // Retrieve the top three most common words and their counts
    $topThreeWords = array_slice($wordCounts, 0, 3);
    
    return $topThreeWords;
}

// Test the function with the example input
$input = "The quick brown fox jumps over the lazy dog. The dog was lazy!";
$output = findTopThreeWords($input);

print_r($output);

?>
