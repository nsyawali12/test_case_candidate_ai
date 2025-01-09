import { Controller, Post, Body, BadRequestException } from '@nestjs/common';
import { IsNumber, Min, IsInt, validate } from 'class-validator';
import { plainToInstance } from 'class-transformer';

class WageCalculationDto {
    @IsNumber()
    @Min(0)
    hourlyWage: number;

    @IsInt()
    @Min(0)
    hoursWorked: number;
}

@Controller('wage')
export class WageController {
    @Post('calculate')
    async calculateWeeklyWage(@Body() wageCalculationDto: WageCalculationDto): Promise<{ weeklyWage: number }> {
        // Validate input data
        const errors = await validate(plainToInstance(WageCalculationDto, wageCalculationDto));
        if (errors.length > 0) {
            throw new BadRequestException('Validation failed');
        }

        const { hourlyWage, hoursWorked } = wageCalculationDto;
        let weeklyWage = 0;

        // Calculate wage based on hours worked and hourly wage
        if (hoursWorked > 40) {
            weeklyWage = (40 * hourlyWage) + ((hoursWorked - 40) * (1.5 * hourlyWage));
        } else {
            weeklyWage = hoursWorked * hourlyWage;
        }

        return { weeklyWage };
    }
}
