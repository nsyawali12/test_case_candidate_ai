// books.controller.ts
import { Controller, Post, Body, Patch, NotFoundException } from '@nestjs/common';
import { BooksService } from './books.service';
import { CreateBookDto, LendBookDto } from './dto/book.dto';

@Controller('books')
export class BooksController {
  constructor(private readonly booksService: BooksService) {}

  @Post()
  addBook(@Body() createBookDto: CreateBookDto): string {
    this.booksService.addBook(createBookDto);
    return 'Book added.';
  }

  @Patch('lend')
  lendBook(@Body() lendBookDto: LendBookDto): string {
    if (!this.booksService.lendBook(lendBookDto)) {
      throw new NotFoundException('Book not available.');
    }
    return 'Book lent.';
  }

  @Patch('return')
  returnBook(@Body() lendBookDto: LendBookDto): string {
    if (!this.booksService.returnBook(lendBookDto)) {
      throw new NotFoundException('Return request unmatched.');
    }
    return 'Book returned.';
  }
}

// books.service.ts
import { Injectable } from '@nestjs/common';
import { CreateBookDto, LendBookDto } from './dto/book.dto';

interface Book {
  title: string;
  author: string;
  isLent: boolean;
  userId?: number;
}

@Injectable()
export class BooksService {
  private books: Book[] = [];

  addBook(createBookDto: CreateBookDto): void {
    this.books.push({ ...createBookDto, isLent: false });
  }

  lendBook(lendBookDto: LendBookDto): boolean {
    const book = this.books.find(b => b.title === lendBookDto.title && !b.isLent);
    if (book) {
      book.isLent = true;
      book.userId = lendBookDto.userId;
      return true;
    }
    return false;
  }

  returnBook(lendBookDto: LendBookDto): boolean {
    const book = this.books.find(b => b.title === lendBookDto.title && b.userId === lendBookDto.userId);
    if (book && book.isLent) {
      book.isLent = false;
      book.userId = undefined;
      return true;
    }
    return false;
  }
}

// dto/book.dto.ts
export class CreateBookDto {
  title: string;
  author: string;
}

export class LendBookDto {
  title: string;
  userId: number;
}

// books.controller.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { BooksController } from './books.controller';
import { BooksService } from './books.service';

describe('BooksController', () => {
  let booksController: BooksController;
  let booksService: BooksService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [BooksController],
      providers: [BooksService],
    }).compile();

    booksController = module.get<BooksController>(BooksController);
    booksService = module.get<BooksService>(BooksService);
  });

  it('should add a book successfully', () => {
    expect(booksController.addBook({ title: 'Clean Code', author: 'Robert C. Martin' })).toBe('Book added.');
  });

  it('should handle lending an available book', () => {
    booksService.addBook({ title: 'Clean Code', author: 'Robert C. Martin' });
    expect(booksController.lendBook({ title: 'Clean Code', userId: 42 })).toBe('Book lent.');
  });

  it('should handle returning a lent book', () => {
    booksService.addBook({ title: 'Clean Code', author: 'Robert C. Martin' });
    booksService.lendBook({ title: 'Clean Code', userId: 42 });
    expect(booksController.returnBook({ title: 'Clean Code', userId: 42 })).toBe('Book returned.');
  });

  it('should throw when lending a non-available book', () => {
    booksService.addBook({ title: 'Clean Code', author: 'Robert C. Martin' });
    booksService.lendBook({ title: 'Clean Code', userId: 42 });
    expect(() => booksController.lendBook({ title: 'Clean Code', userId: 99 })).toThrow('Book not available.');
  });

  it('should throw when returning an unmatched book', () => {
    expect(() => booksController.returnBook({ title: 'Some Other Book', userId: 999 })).toThrow('Return request unmatched.');
  });
});
