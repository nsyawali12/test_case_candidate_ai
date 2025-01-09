<?php
    function mostCommonWords($input) {
        // lowercase the input and remove punctuation
        $input = strtolower(preg_replace('#[^\w\s]#', '', $input));
        // explode input into array of words
        $words = explode(' ', $input);
        // create an empty associative array to keep count of word frequency
        $frequency = array();
        foreach ($words as $word) {
            if (!isset($frequency[$word])) {
                $frequency[$word] = 1; // initialize word frequency to 1 if it doesn't exist
            } else {
                $frequency[$word]++; // increment word frequency if the word already exists
            }
        }
        // sort the array in descending order by frequency
        arsort($frequency);
        // slice the array to get top 3
        $frequency = array_slice($frequency, 0, 3);
        // formating output as tuples
        $output = [];
        foreach($frequency as $word => $count) {
            $output[] = array($word, $count);
        }
        return $output;
    }

    $input = "The quick brown fox jumps over the lazy dog. The dog was lazy!";
    $result = mostCommonWords($input);
    print_r($result);
?>
