<?php

// Function to get the top three most common words from given string
function getTopThreeWords($text) {
    // Remove punctuation and convert to lowercase for uniformity
    $text = strtolower(preg_replace("/[^\w\s]/", '', $text));
    
    // Split string into words
    $words = explode(' ', $text);
    
    // Initialize an array to count word occurrences
    $wordCounts = array();
    
    // Count each word's frequency
    foreach ($words as $word) {
        if ($word != '') {
            // Increment the count of each word
            if (isset($wordCounts[$word])) {
                $wordCounts[$word]++;
            } else {
                $wordCounts[$word] = 1;
            }
        }
    }
    
    // Sort the words based on frequency (and by key as a tie-breaker)
    arsort($wordCounts);
    
    // Return the top 3 most frequent words
    return array_slice($wordCounts, 0, 3);
}

// Usage
$input = "The quick brown fox jumps over the lazy dog. The dog was lazy!";
$result = getTopThreeWords($input);

// Output in the desired format
$output = array();
foreach ($result as $word => $freq) {
    $output[] = array($word, $freq);
}

print_r($output);
?>
