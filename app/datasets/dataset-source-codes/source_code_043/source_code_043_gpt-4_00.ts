import { Body, Controller, Post } from '@nestjs/common';
import { IsInt, IsNotEmpty, IsPositive, Min } from 'class-validator';

class WageInput {
  @IsNotEmpty()
  @IsPositive()
  hourlyWage: number;

  @IsNotEmpty()
  @Min(0)
  @IsInt()
  hoursWorked: number;
}

class WageOutput {
  weeklyWage: number;
}

@Controller('wage')
export class WageController {
  @Post('calculate')
  calculate(@Body() wageInput: WageInput): WageOutput {

    let overTime = 0
    let weeklyWage = 0
    // Calculate the number of overtime hours, if any
    if(wageInput.hoursWorked > 40) {
      overTime = wageInput.hoursWorked - 40
    }

    // Calculate the weekly salary considering the overtime hours
    weeklyWage = (wageInput.hourlyWage * Math.min(40, wageInput.hoursWorked)) + (overTime * wageInput.hourlyWage * 1.5)
    
    return {
      weeklyWage,  
    };
  }
}
