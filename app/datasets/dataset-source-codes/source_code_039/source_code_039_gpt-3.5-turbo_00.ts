// book.dto.ts
export class BookDTO {
  readonly title: string;
  readonly author: string;
}

export class LendBookDTO {
  readonly title: string;
  readonly userId: number;
}
