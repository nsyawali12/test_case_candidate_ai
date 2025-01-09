<?php

function countTopWords($inputString) {
    // Convert the input string to lowercase and remove punctuation
    $cleanString = preg_replace("/[\W_]+/", " ", strtolower($inputString));

    // Split the string into an array of words
    $words = explode(" ", $cleanString);

    // Count the frequency of each word
    $wordCount = array_count_values($words);

    // Sort the words by frequency in descending order
    arsort($wordCount);

    // Get the top three most common words
    $topWords = array_slice($wordCount, 0, 3);

    // Format the output as an array of tuples
    $output = [];
    foreach ($topWords as $word => $count) {
        $output[] = [$word, $count];
    }

    return $output;
}

// Test the function with the example input
$inputString = "The quick brown fox jumps over the lazy dog. The dog was lazy!";
$output = countTopWords($inputString);
print_r($output);

?>
