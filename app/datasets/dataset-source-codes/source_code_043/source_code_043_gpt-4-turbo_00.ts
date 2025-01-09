import { Controller, Post, Body, BadRequestException } from '@nestjs/common';

class CalculateWageDto {
    hourlyWage: number;
    hoursWorked: number;
}

@Controller('wages')
export class WageController {
    @Post('calculate')
    calculateWeeklyWage(@Body() { hourlyWage, hoursWorked }: CalculateWageDto): any {
        // Validate that hourlyWage and hoursWorked are positive numbers
        if (hourlyWage <= 0 || hoursWorked < 0 || !Number.isInteger(hoursWorked)) {
            throw new BadRequestException('Invalid input: hourlyWage must be a positive number and hoursWorked must be a positive integer.');
        }

        let weeklyWage: number;
        if (hoursWorked <= 40) {
            // Calculate weekly wage without overtime
            weeklyWage = hoursWorked * hourlyWage;
        } else {
            // Calculate weekly wage with overtime
            const regularWage = 40 * hourlyWage;
            const overtimeHours = hoursWorked - 40;
            const overtimeWage = overtimeHours * (1.5 * hourlyWage);
            weeklyWage = regularWage + overtimeWage;
        }

        return { weeklyWage };
    }
}
