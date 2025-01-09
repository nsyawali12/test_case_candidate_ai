import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'orderBy'
})
export class OrderByPipe implements PipeTransform {

  // Implement the transform method which is called for each value in the pipe
  transform(array: any[], key: string, order: string = 'asc'): any[] {

    // Error handling for invalid input data
    if (!Array.isArray(array) || !key || (order !== 'asc' && order !== 'desc')) {
      throw new Error(`Invalid input data. Array: ${Array.isArray(array)}, Key: ${key}, Order: ${order}`);
    }

    // The compare function to be used in the sort method
    const compare = (a, b) => {
      if (!a.hasOwnProperty(key) || !b.hasOwnProperty(key)) {
        return 0; 
      }

      const varA = (typeof a[key] === 'string') ? a[key].toUpperCase() : a[key];
      const varB = (typeof b[key] === 'string') ? b[key].toUpperCase() : b[key];

      let comparison = 0;
      if (varA > varB) {
        comparison = 1;
      } else if (varA < varB) {
        comparison = -1;
      }
      return comparison;
    }

    // Sort the array in ascending or descending order
    return array.sort((a, b) => {
      return order === 'asc' ? compare(a, b) : -compare(a, b);
    });
  }
}
