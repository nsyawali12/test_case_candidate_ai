use Illuminate\Support\Collection;

// Define the custom collection pipeline
Collection::macro('customPipeline', function () {
    return $this
        ->filter(function ($value) {
            return $value >= 10;
        })
        ->map(function ($value) {
            return $value ** 2;
        })
        ->average();
});

// Input data
$input = collect([2, 10, 3, 4, 15, 20]);

// Applying the custom pipeline and calculating the average
$result = $input->customPipeline();

echo $result;
