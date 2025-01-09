// book.dto.ts
export class BookDto {
  title: string;
  author: string;
}

export class LendBookDto {
  title: string;
  userId: number;
}

// book.service.ts
import { Injectable } from '@nestjs/common';

@Injectable()
export class BookService {
  private books: Map<string, boolean> = new Map();

  addBook(bookDto: BookDto) {
    this.books.set(bookDto.title, true);
    return 'Book added.';
  }

  lendBook(lendBookDto: LendBookDto) {
    if (this.books.has(lendBookDto.title) && this.books.get(lendBookDto.title)) {
      this.books.set(lendBookDto.title, false);
      return 'Book lent.';
    } else {
      throw new Error('Book not available for lending.');
    }
  }

  returnBook(lendBookDto: LendBookDto) {
    if (this.books.has(lendBookDto.title) && !this.books.get(lendBookDto.title)) {
      this.books.set(lendBookDto.title, true);
      return 'Book returned.';
    } else {
      throw new Error('Book not available for returning.');
    }
  }
}

// book.controller.ts
import { Controller, Post, Body } from '@nestjs/common';
import { BookService } from './book.service';
import { BookDto, LendBookDto } from './book.dto';

@Controller('books')
export class BookController {
  constructor(private readonly bookService: BookService) {}

  @Post('add')
  addBook(@Body() bookDto: BookDto) {
    return this.bookService.addBook(bookDto);
  }

  @Post('lend')
  lendBook(@Body() lendBookDto: LendBookDto) {
    return this.bookService.lendBook(lendBookDto);
  }

  @Post('return')
  returnBook(@Body() lendBookDto: LendBookDto) {
    return this.bookService.returnBook(lendBookDto);
  }
}

// End of the code
