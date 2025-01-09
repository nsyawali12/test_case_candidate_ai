$collection = collect([2, 10, 3, 4, 15, 20]);

$result = $collection
    ->filter(function ($value) { // Filter out numbers less than 10
        return $value >= 10;
    })
    ->map(function ($value) { // Calculate the square of remaining numbers
        return $value ** 2;
    })
    ->avg(); // Compute the average of the squared numbers

echo $result; // Outputs the average
