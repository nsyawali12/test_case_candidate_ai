import { Body, Controller, Post } from '@nestjs/common';
import { IsNumber, IsInt, Min } from 'class-validator';

class CalculateWageDto {
  @IsNumber()
  @Min(0)
  hourlyWage: number;

  @IsInt()
  @Min(0)
  hoursWorked: number;
}

@Controller()
export class AppController {
  @Post('calculateWage')
  calculateWage(@Body() body: CalculateWageDto) {
    const { hourlyWage, hoursWorked } = body;

    let weeklyWage = 0;

    if (hoursWorked > 40) {
      const regularHours = 40;
      const overtimeHours = hoursWorked - regularHours;
      weeklyWage = (regularHours * hourlyWage) + (overtimeHours * (1.5 * hourlyWage));
    } else {
      weeklyWage = hoursWorked * hourlyWage;
    }

    return { weeklyWage };
  }
}
