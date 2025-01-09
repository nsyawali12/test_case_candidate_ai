import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'orderBy'
})
export class OrderByPipe implements PipeTransform {
  transform(array: any[], key: string, order: string): any[] {
    if (!Array.isArray(array) || !key || (order !== 'asc' && order !== 'desc')) {
      console.error('Invalid input data');
      return array;
    }

    const compareFn = (a, b) => {
      if (a[key] < b[key]) {
        return order === 'asc' ? -1 : 1;
      }
      if (a[key] > b[key]) {
        return order === 'asc' ? 1 : -1;
      }
      return 0;
    };

    return array.slice().sort(compareFn);
  }
}
