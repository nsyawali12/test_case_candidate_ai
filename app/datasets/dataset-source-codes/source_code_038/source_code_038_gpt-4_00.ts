// Import necessary decorators, services and modules from NestJS
import { Body, Controller, Get, Param, Post, Put, NotFoundException, BadRequestException } from "@nestjs/common";
import { InjectRepository } from "@nestjs/typeorm";
import { Repository } from "typeorm";
import { Product } from "./product.entity"; // Entity for Product

// Define a DTO class to manage the data transferred when adding or updating a product
export class ProductDto {
  id: string;
  name: string;
  description: string;
  price: number;
  quantity: number;
}

@Controller('product')
export class ProductController {
  constructor(
    @InjectRepository(Product)
    private readonly productRepository: Repository<Product>,
  ) {}

  @Get()
  async findAll(): Promise<Product[]> {
    return this.productRepository.find();
  }

  @Post()
  async create(@Body() productDto: ProductDto): Promise<Product> {
    const product = new Product();
    product.id = productDto.id;
    product.name = productDto.name;
    product.description = productDto.description;
    product.price = productDto.price;
    product.quantity = productDto.quantity;

    return this.productRepository.save(product);
  }

  @Put(':id')
  async update(
    @Param('id') id: string,
    @Body() productDto: ProductDto,
  ): Promise<void> {
    const product = await this.productRepository.findOne(id);

    if(!product){
      throw new NotFoundException('Product not found');
    }

    if(productDto.quantity < 0){
      throw new BadRequestException('Quantity should be a positive number');
    }

    product.name = productDto.name;
    product.description = productDto.description;
    product.price = productDto.price;
    product.quantity = productDto.quantity;

    await this.productRepository.save(product);
  }

  @Get(':id')
  async findOne(@Param('id') id: string): Promise<Product> {
    const product = await this.productRepository.findOne(id);

    if (!product) {
      throw new NotFoundException('Product not found');
    }

    return product;
  }
}
