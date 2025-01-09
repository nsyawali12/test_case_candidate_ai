$result = collect([2, 10, 3, 4, 15, 20])
    ->filter(function ($value) {
        return $value >= 10; // Filter out numbers less than 10
    })
    ->map(function ($value) {
        return pow($value, 2); // Calculate the square of the remaining numbers
    })
    ->average(); // Compute the average of the squared numbers

echo $result; // Output the result
