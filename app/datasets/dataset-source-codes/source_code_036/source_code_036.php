$input->filter(function ($number) 
        return $number >= 10;
    })
    ->map(function ($number) {
        return $number ** 2;
    })
    ->average();