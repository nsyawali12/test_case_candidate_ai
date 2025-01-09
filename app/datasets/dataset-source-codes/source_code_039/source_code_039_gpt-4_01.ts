// Import necessary dependencies
import { Controller, Post, Body, Get, Patch, NotFoundException, Param } from '@nestjs/common';
import { BooksService } from './books.service';

// Define DTOs
class CreateBookDto {
    title: string;
    author: string;
}

class LendBookDto {
    title: string;
    userId: number;
}

// Create the Books controller with necessary endpoints and HTTP methods
@Controller('books')
export class BooksController {
    constructor(private readonly booksService: BooksService) {}

    @Post()
    addBook(@Body() createBookDto: CreateBookDto): string {
        return this.booksService.addBook(createBookDto);
    }

    @Patch('/lend')
    lendBook(@Body() lendBookDto: LendBookDto): string {
        const res = this.booksService.lendBook(lendBookDto);

        if (!res) {
            throw new NotFoundException('Book not available');
        }

        return res;
    }

    @Patch('/return/:title')
    returnBook(@Param('title') title: string, @Body() lendBookDto: LendBookDto): string {
        const res = this.booksService.returnBook(title, lendBookDto);

        if (!res) {
            throw new NotFoundException('Return request not valid');
        }

        return res;
    }
}


// Define the Books service with necessary functionality for lending and managing books
import { Injectable } from '@nestjs/common';

// Define a Book interface for handling book objects within the service
interface Book {
    title: string;
    author: string;
    isBorrowed: boolean;
    borrowedBy?: number;
}

@Injectable()
export class BooksService {
    private books: Book[] = [];

    addBook(createBookDto: CreateBookDto): string {
        this.books.push({...createBookDto, isBorrowed: false });
        return 'Book added';
    }

    lendBook(lendBookDto: LendBookDto): string {
        const book = this.books.find(b => b.title === lendBookDto.title && !b.isBorrowed);

        if (!book) return null;

        book.isBorrowed = true;
        book.borrowedBy = lendBookDto.userId;

        return 'Book lent';
    }

    returnBook(title: string, lendBookDto: LendBookDto): string {
        const book = this.books.find(b => b.title === title && b.isBorrowed && b.borrowedBy === lendBookDto.userId);

        if (!book) return null;

        book.isBorrowed = false;
        book.borrowedBy = null;

        return 'Book returned';
    }
}
