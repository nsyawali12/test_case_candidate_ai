// book.dto.ts
export class BookDto {
  title: string;
  author: string;
}

export class LendBookDto {
  title: string;
  userId: number;
}


// ======================= service

// book.service.ts
import { Injectable, NotFoundException, ConflictException } from '@nestjs/common';
import { BookDto, LendBookDto } from './book.dto';

@Injectable()
export class BookService {
  private books: BookDto[] = [];

  addBook(bookDto: BookDto): void {
    this.books.push(bookDto);
  }

  lendBook(lendBookDto: LendBookDto): void {
    const book = this.books.find(b => b.title === lendBookDto.title);
    if (!book) {
      throw new NotFoundException('Book not found');
    }

    if (!book.available) {
      throw new ConflictException('Book is not available');
    }
    
    book.available = false;
  }

  returnBook(lendBookDto: LendBookDto): void {
    const book = this.books.find(b => b.title === lendBookDto.title);
    if (!book) {
      throw new NotFoundException('Book not found');
    }
    book.available = true;
  }
}

// ======================= controller

// book.controller.ts
import { Controller, Post, Body, NotFoundException, ConflictException } from '@nestjs/common';
import { BookService } from './book.service';
import { BookDto, LendBookDto } from './book.dto';

@Controller('books')
export class BookController {
  constructor(private readonly bookService: BookService) {}

  @Post('add')
  addBook(@Body() bookDto: BookDto): string {
    this.bookService.addBook(bookDto);
    return 'Book added';
  }

  @Post('lend')
  lendBook(@Body() lendBookDto: LendBookDto): string {
    try {
      this.bookService.lendBook(lendBookDto);
      return 'Book lent';
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw new NotFoundException(error.message);
      } else if (error instanceof ConflictException) {
        throw new ConflictException(error.message);
      }
      throw new Error('Internal server error');
    }
  }

  @Post('return')
  returnBook(@Body() lendBookDto: LendBookDto): string {
    try {
      this.bookService.returnBook(lendBookDto);
      return 'Book returned';
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw new NotFoundException(error.message);
      }
      throw new Error('Internal server error');
    }
  }
}