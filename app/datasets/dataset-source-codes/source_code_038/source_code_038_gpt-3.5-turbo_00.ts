// inventory.dto.ts
import { IsInt, IsNotEmpty, IsNumber, IsString, Min } from 'class-validator';

export class ProductDto {
  @IsString()
  @IsNotEmpty()
  id: string;

  @IsString()
  @IsNotEmpty()
  name: string;

  @IsString()
  description: string;

  @IsNumber()
  @IsNotEmpty()
  price: number;

  @IsInt()
  @Min(0)
  @IsNotEmpty()
  availableQuantity: number;
}

// inventory.interface.ts
export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  availableQuantity: number;
}

// inventory.module.ts
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { InventoryController } from './inventory.controller';
import { InventoryService } from './inventory.service';
import { Product } from './product.entity';

@Module({
  imports: [TypeOrmModule.forFeature([Product])],
  controllers: [InventoryController],
  providers: [InventoryService]
})
export class InventoryModule {}  

// product.entity.ts
import { Entity, Column, PrimaryGeneratedColumn } from 'typeorm';

@Entity()
export class Product {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @Column()
  description: string;

  @Column('decimal')
  price: number;

  @Column()
  availableQuantity: number;
}

// inventory.controller.ts
import { Controller, Get, Post, Put, Body, Param } from '@nestjs/common';
import { InventoryService } from './inventory.service';
import { ProductDto } from './inventory.dto';

@Controller('inventory')
export class InventoryController {
  constructor(private readonly inventoryService: InventoryService) {}

  @Post('add-product')
  async addProduct(@Body() productDto: ProductDto) {
    return this.inventoryService.addProduct(productDto);
  }

  @Get('products')
  async getProducts() {
    return this.inventoryService.getProducts();
  }

  @Put('update-quantity/:id')
  async updateQuantity(@Param('id') id: string, @Body('quantity') quantity: number) {
    return this.inventoryService.updateQuantity(id, quantity);
  }
}

// inventory.service.ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Product } from './product.entity';
import { ProductDto } from './inventory.dto';

@Injectable()
export class InventoryService {
  constructor(
    @InjectRepository(Product)
    private productRepository: Repository<Product>,
  ) {}

  async addProduct(productDto: ProductDto): Promise<Product> {
    const newProduct = this.productRepository.create(productDto);
    return this.productRepository.save(newProduct);
  }

  async getProducts(): Promise<Product[]> {
    return this.productRepository.find();
  }

  async updateQuantity(id: string, quantity: number): Promise<Product> {
    const product = await this.productRepository.findOne(id);
    if (!product) {
      throw new Error('Product not found');
    }

    product.availableQuantity = quantity;
    return this.productRepository.save(product);
  }
}
