<?php

function getTopThreeWords($text) {
    // Normalize the text: lower case and remove punctuation
    $normalizedText = strtolower(preg_replace("/[^\w\s]/", "", $text));

    // Split the text into words
    $words = explode(" ", $normalizedText);
    
    // Count the frequency of each word
    $wordCount = array_count_values($words);

    // Sort the array based on frequency in descending order
    arsort($wordCount);

    // Get the top three most frequent words
    $topThree = array_slice($wordCount, 0, 3);
    
    // Format the output to show word and count
    $result = [];
    foreach ($topThree as $word => $count) {
        $result[] = [$word, $count];
    }
    
    return $result;
}

// Example usage
$input = "The quick brown fox jumps over the lazy dog. The dog was lazy!";
$output = getTopThreeWords($input);
print_r($output);
?>
