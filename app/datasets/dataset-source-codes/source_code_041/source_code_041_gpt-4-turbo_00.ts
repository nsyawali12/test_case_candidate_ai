import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'orderBy'
})
export class OrderByPipe implements PipeTransform {

  transform(array: any[], key: string, order: string = 'asc'): any[] {
    if (!Array.isArray(array)) {
      throw new Error('OrderByPipe: input is not an array');
    }

    if (!key) {
      throw new Error('OrderByPipe: \'key\' is required');
    }

    // Sort the array by specified object key, handle both string and number comparisons
    return array.sort((a, b) => {
      if (!a.hasOwnProperty(key) || !b.hasOwnProperty(key)) {
        throw new Error(`OrderByPipe: key '${key}' not found in one or more items`);
      }

      const varA = (typeof a[key] === 'string') ? a[key].toUpperCase() : a[key];
      const varB = (typeof b[key] === 'string') ? b[key].toUpperCase() : b[key];

      let comparison = 0;
      if (varA > varB) {
        comparison = 1;
      } else if (varA < varB) {
        comparison = -1;
      }

      return (order === 'desc') ? (comparison * -1) : comparison;
    });
  }

}
