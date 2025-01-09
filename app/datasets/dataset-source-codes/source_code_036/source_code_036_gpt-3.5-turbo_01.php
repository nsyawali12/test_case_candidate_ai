use Illuminate\Support\Collection;

$numbers = collect([2, 10, 3, 4, 15, 20]);

$result = $numbers
    ->filter(function ($number) {
        return $number >= 10;
    })
    ->map(function ($number) {
        return pow($number, 2);
    })
    ->average();

return $result;
