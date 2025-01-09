// book.controller.ts
import { Body, Controller, Get, Post, Put, Res, HttpStatus, NotFoundException } from '@nestjs/common';
import { BookService } from './book.service';
import { CreateBookDto } from './dto/create-book.dto';
import { LendBookDto } from './dto/lend-book.dto';
import { ReturnBookDto } from './dto/return-book.dto';

@Controller('books')
export class BookController {
  constructor(private readonly bookService: BookService) {}

  @Post('add')
  async addBook(@Res() res, @Body() createBookDto: CreateBookDto) {
    const book = await this.bookService.addBook(createBookDto);
    return res.status(HttpStatus.CREATED).json(book);
  }

  @Put('lend')
  async lendBook(@Res() res, @Body() lendBookDto: LendBookDto) {
    const response = await this.bookService.lendBook(lendBookDto);
    return res.status(HttpStatus.OK).json(response);
  }

  @Put('return')
  async returnBook(@Res() res, @Body() returnBookDto: ReturnBookDto) {
    const response = await this.bookService.returnBook(returnBookDto);
    return res.status(HttpStatus.OK).json(response);
  }
}

// book.service.ts
import { Injectable, NotFoundException } from '@nestjs/common';
import { CreateBookDto } from './dto/create-book.dto';
import { LendBookDto } from './dto/lend-book.dto';
import { ReturnBookDto } from './dto/return-book.dto';

@Injectable()
export class BookService {
  private books = [];

  addBook(createBookDto: CreateBookDto){
    const newBook = { ...createBookDto, status: 'available' };
    this.books.push(newBook);
    return { message: 'Book added.' };
  }

  lendBook(lendBookDto: LendBookDto){
    const bookIndex = this.books.findIndex((book) => book.title === lendBookDto.title);
    if(bookIndex === -1 || this.books[bookIndex].status !== 'available') {
      throw new NotFoundException(`The book ${lendBookDto.title} is unavailable.`);
    }
    this.books[bookIndex].status = 'lent';
    return { message: 'Book lent.' };
  }

  returnBook(returnBookDto: ReturnBookDto){
    const bookIndex = this.books.findIndex((book) => book.title === returnBookDto.title && book.status === 'lent');
    if(bookIndex === -1){
      throw new NotFoundException(`The book ${returnBookDto.title} is not lent.`);
    }
    this.books[bookIndex].status = 'available';
    return { message: 'Book returned.' };
  }
}
