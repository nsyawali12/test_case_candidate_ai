// wage-input.dto.ts
import { IsNumber, IsPositive } from 'class-validator';

export class WageInputDto {
  @IsNumber()
  @IsPositive()
  hourlyWage: number;

  @IsNumber()
  @IsPositive()
  hoursWorked: number;
}

// It will handle the input validation too.


// employee.controller.ts
import { Body, Controller, Post } from '@nestjs/common';
import { WageInputDto } from './dto/wage-input.dto';
import { EmployeeService } from './employee.service';

@Controller('employee')
export class EmployeeController {
  constructor(private readonly employeeService: EmployeeService) {}

  @Post('/wage')
  async getWage(@Body() input: WageInputDto) {
    return this.employeeService.getWage(input);
  }
}

// employee.service.ts
import { Injectable } from '@nestjs/common';
import { WageInputDto } from './dto/wage-input.dto';

const WEEKLY_HOURS = 40;

@Injectable()
export class EmployeeService {
  async getWage(input: WageInputDto) {
    let weeklyHours = 0;
    let overTimeHours = 0;
    let weeklyWage = 0;

    const hasDoneOverTime = input.hoursWorked > WEEKLY_HOURS;

    if (hasDoneOverTime) {
      weeklyHours = WEEKLY_HOURS;
      overTimeHours = input.hoursWorked - WEEKLY_HOURS;
    } else {
      weeklyHours = input.hoursWorked;
    }

    weeklyWage = weeklyHours * input.hourlyWage;

    if (hasDoneOverTime) {
      weeklyWage = weeklyWage + overTimeHours * (input.hourlyWage * 1.5);
    }

    return { weeklyWage };
  }
}