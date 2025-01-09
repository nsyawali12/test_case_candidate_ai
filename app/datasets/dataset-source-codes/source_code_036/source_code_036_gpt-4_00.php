<?php

use Illuminate\Support\Collection;

function calculateAverageSquared(Collection $numbers) {
    return $numbers
        // Filter out numbers less than 10
        ->filter(function ($number) {
            return $number >= 10;
        })
        // Calculate the square of the remaining numbers
        ->map(function ($number) {
            return pow($number, 2);
        })
        // Compute the average of the squared numbers
        ->avg();
}

// Test the function with the provided input
echo calculateAverageSquared(collect([2, 10, 3, 4, 15, 20]));

?>
