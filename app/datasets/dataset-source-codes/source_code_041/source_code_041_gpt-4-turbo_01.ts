import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'orderBy'
})
export class OrderByPipe implements PipeTransform {

  transform(value: any[], key: string, order: string = 'asc'): any[] {
    if (!Array.isArray(value)) {
      throw new Error('Provided value must be an array');
    }
    if (value.length === 0) {
      return value;
    }
    if (typeof value[0][key] === 'undefined') {
      throw new Error(`Key '${key}' does not exist on array items`);
    }

    const sortedArray = [...value].sort((a, b) => {
      if (a[key] < b[key]) {
        return order === 'desc' ? 1 : -1;
      } else if (a[key] > b[key]) {
        return order === 'desc' ? -1 : 1;
      } else {
        return 0;
      }
    });

    return sortedArray;
  }
}
