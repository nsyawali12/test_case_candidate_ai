<?php

use Illuminate\Support\Collection;

function processNumbers(Collection $numbers) {
    return $numbers
        ->filter(function ($number) { // Filter out numbers less than 10
            return $number >= 10;
        })
        ->map(function ($number) { // Calculate the square of remaining numbers
            return pow($number, 2);
        })
        ->avg(); // Compute the average of the squared numbers
}

// Use the function
$input = collect([2, 10, 3, 4, 15, 20]);
$output = processNumbers($input);

// Output: 306.66666666667
echo $output;

?>
