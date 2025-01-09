// books.controller.ts
import { Body, Controller, HttpException, HttpStatus, Post } from '@nestjs/common';
import { BooksService } from './books.service';
import { CreateBookDto, LendBookDto, ReturnBookDto } from './dto/books.dto';

@Controller('books')
export class BooksController {
  constructor(private readonly booksService: BooksService) {}

  @Post('add')
  addBook(@Body() createBookDto: CreateBookDto): string {
    this.booksService.addBook(createBookDto);
    return 'Book added.';
  }

  @Post('lend')
  lendBook(@Body() lendBookDto: LendBookDto): string {
    const result = this.booksService.lendBook(lendBookDto);
    if (!result) {
      throw new HttpException('Book not available', HttpStatus.BAD_REQUEST);
    }
    return 'Book lent.';
  }

  @Post('return')
  returnBook(@Body() returnBookDto: ReturnBookDto): string {
    const result = this.booksService.returnBook(returnBookDto);
    if (!result) {
      throw new HttpException('Book not found or not lent', HttpStatus.BAD_REQUEST);
    }
    return 'Book returned.';
  }
}

// books.service.ts
import { Injectable } from '@nestjs/common';
import { CreateBookDto, LendBookDto, ReturnBookDto } from './dto/books.dto';

interface Book {
  title: string;
  author: string;
  isAvailable: boolean;
  userId?: number;
}

@Injectable()
export class BooksService {
  private books: Book[] = [];

  addBook(createBookDto: CreateBookDto) {
    this.books.push({ ...createBookDto, isAvailable: true });
  }

  lendBook(lendBookDto: LendBookDto): boolean {
    const book = this.books.find(b => b.title === lendBookDto.title && b.isAvailable);
    if (book) {
      book.isAvailable = false;
      book.userId = lendBookDto.userId;
      return true;
    }
    return false;
  }

  returnBook(returnBookDto: ReturnBookDto): boolean {
    const book = this.books.find(b => b.title === returnBookDto.title && b.userId === returnBookDto.userId);
    if (book && !book.isAvailable) {
      book.isAvailable = true;
      book.userId = undefined;
      return true;
    }
    return false;
  }
}

// dto/books.dto.ts
export class CreateBookDto {
  title: string;
  author: string;
}

export class LendBookDto {
  title: string;
  userId: number;
}

export class ReturnBookDto {
  title: string;
  userId: number;
}

// books.controller.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { BooksController } from './books.controller';
import { BooksService } from './books.service';

describe('BooksController', () => {
  let booksController: BooksController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [BooksController],
      providers: [BooksService],
    }).compile();

    booksController = module.get<BooksController>(BooksController);
  });

  it('should add a book correctly', () => {
    expect(booksController.addBook({ title: 'Test Book', author: 'Tester' })).toBe('Book added.');
  });

  it('should handle lending of an available book', () => {
    booksController.addBook({ title: 'Test Book', author: 'Tester' });
    expect(booksController.lendBook({ title: 'Test Book', userId: 1 })).toBe('Book lent.');
  });

  it('should handle returning of a lent book', () => {
    booksController.addBook({ title: 'Test Book', author: 'Tester' });
    booksController.lendBook({ title: 'Test Book', userId: 1 });
    expect(booksController.returnBook({ title: 'Test Book', userId: 1 })).toBe('Book returned.');
  });

  it('should throw error when lending a non-available book', () => {
    booksController.addBook({ title: 'Test Book', author: 'Tester' });
    booksController.lendBook({ title: 'Test Book', userId: 1 });
    expect(() => booksController.lendBook({ title: 'Test Book', userId: 2 })).toThrowError(new HttpException('Book not available', HttpStatus.BAD_REQUEST));
  });

  it('should throw error when returning a non-lent or unmatched book', () => {
    booksController.addBook({ title: 'Test Book', author: 'Tester' });
    expect(() => booksController.returnBook({ title: 'Test Book', userId: 1 })).toThrowError(new HttpException('Book not found or not lent', HttpStatus.BAD_REQUEST));
  });
});
