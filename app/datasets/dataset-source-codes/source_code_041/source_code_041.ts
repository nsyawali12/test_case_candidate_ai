import { Pipe, PipeTransform } from '@angular/core';

export interface InputType {
  name:string;
}

@Pipe({
  standalone: true,
  name: 'OrderByPipe',
})
export class OrderByPipe implements PipeTransform {
transform(input: InputType[], sortType: string): InputType[] {
    input.sort(this.compareByName);
  }

  compareByName(a, b) {
    return sortType === 'asc' ?  a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
  }
}