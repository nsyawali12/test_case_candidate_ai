import { Controller, Post, Body, HttpException, HttpStatus } from '@nestjs/common';

@Controller('calculate-wage')
export class WageController {
  
  @Post()
  calculateWeeklyWage(@Body() data: { hourlyWage: number, hoursWorked: number }): { weeklyWage: number } {
    const { hourlyWage, hoursWorked } = data;

    if (hourlyWage <= 0 || !Number.isInteger(hoursWorked) || hoursWorked <= 0) {
      throw new HttpException('Invalid input data', HttpStatus.BAD_REQUEST);
    }

    let weeklyWage: number;
    if (hoursWorked <= 40) {
      weeklyWage = hoursWorked * hourlyWage;
    } else {
      weeklyWage = (40 * hourlyWage) + ((hoursWorked - 40) * (1.5 * hourlyWage));
    }

    return { weeklyWage };
  }
}
