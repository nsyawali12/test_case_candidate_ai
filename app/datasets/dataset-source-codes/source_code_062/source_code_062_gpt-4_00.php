function findTopWords($input) {
    // Remove punctuation and convert string to lowercase
    $input = strtolower(preg_replace("/[^a-z0-9\s]/", '', $input));

    // Create an array from the input string
    $words = explode(' ', $input);

    // Use array_count_values to get frequencies
    $frequencies = array_count_values($words);
    
    // Use arsort to sort in descending order while maintaining index association
    arsort($frequencies);

    // Use array_slice to get the top 3
    $top_three = array_slice($frequencies, 0, 3, true);

    // Convert to format [("word", count), ...]
    $output = [];
    foreach($top_three as $word => $count) {
        $output[] = array($word, $count);
    }

    return $output;
}

// Test with given input
$input = "The quick brown fox jumps over the lazy dog. The dog was lazy!";
print_r(findTopWords($input));
