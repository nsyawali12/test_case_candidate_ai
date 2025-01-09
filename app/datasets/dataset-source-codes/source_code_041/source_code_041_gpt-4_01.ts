import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'orderBy'
})
export class OrderByPipe implements PipeTransform {

  transform(array: any[], key: string, order = 'asc'): any[] {
    // Handle error for invalid input data
    if (!Array.isArray(array) || !key || (order !== 'asc' && order !== 'desc')) {
      console.error('Invalid input data');
      return;
    }
    const result = array.sort((a, b) => {
          // Check if property exists on object
          if (!a.hasOwnProperty(key) || !b.hasOwnProperty(key)) {
            return 0;
          }

          let comparison = 0;
          if (a[key] > b[key]) {
            comparison = 1;
          } else if (a[key] < b[key]) {
            comparison = -1;
          }

          // Handle sorting order
          return (order === 'desc') ? (comparison * -1) : comparison;
    });

    return result; 
  }

}
