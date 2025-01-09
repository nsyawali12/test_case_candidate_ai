import { Controller, Post, Body, HttpException, HttpStatus } from '@nestjs/common';

interface WeeklyWageInput {
  hourlyWage: number;
  hoursWorked: number;
}

@Controller('calculate-weekly-wage')
export class WeeklyWageController {
  @Post()
  calculateWeeklyWage(@Body() data: WeeklyWageInput): { weeklyWage: number } {
    // Input validation
    if (data.hourlyWage <= 0 || data.hoursWorked <= 0 || !Number.isInteger(data.hoursWorked)) {
      throw new HttpException('Invalid input. Hourly wage must be positive and hours worked must be a positive integer', HttpStatus.BAD_REQUEST);
    }

    const regularHours = Math.min(data.hoursWorked, 40);
    const overtimeHours = Math.max(data.hoursWorked - 40, 0);

    const weeklyWage = (regularHours * data.hourlyWage) + (overtimeHours * (1.5 * data.hourlyWage));

    return { weeklyWage };
  }
}
