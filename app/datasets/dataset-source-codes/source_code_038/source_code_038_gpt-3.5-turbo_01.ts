// inventory.dto.ts
import { IsString, IsNumber, IsPositive } from 'class-validator';

export class AddProductDto {
  @IsString()
  name: string;

  @IsString()
  description: string;

  @IsNumber()
  price: number;

  @IsNumber()
  @IsPositive()
  quantity: number;
}

export class UpdateProductQuantityDto {
  @IsNumber()
  @IsPositive()
  quantity: number;
}

// inventory.interface.ts
export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  quantity: number;
}

// inventory.module.ts
import { Module } from '@nestjs/common';
import { InventoryController } from './inventory.controller';
import { InventoryService } from './inventory.service';

@Module({
  controllers: [InventoryController],
  providers: [InventoryService],
})
export class InventoryModule {}

// inventory.controller.ts
import { Controller, Post, Get, Put, Body, Param } from '@nestjs/common';
import { InventoryService } from './inventory.service';
import { AddProductDto, UpdateProductQuantityDto } from './inventory.dto';
import { Product } from './inventory.interface';

@Controller('inventory')
export class InventoryController {
  constructor(private readonly inventoryService: InventoryService) {}

  @Post()
  async addProduct(@Body() addProductDto: AddProductDto): Promise<Product> {
    return this.inventoryService.addProduct(addProductDto);
  }

  @Get()
  async getProducts(): Promise<Product[]> {
    return this.inventoryService.getProducts();
  }

  @Put(':id')
  async updateProductQuantity(
    @Param('id') id: string,
    @Body() updateProductQuantityDto: UpdateProductQuantityDto,
  ): Promise<Product> {
    return this.inventoryService.updateProductQuantity(id, updateProductQuantityDto);
  }
}

// inventory.service.ts
import { Injectable } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { AddProductDto, UpdateProductQuantityDto } from './inventory.dto';
import { Product } from './inventory.interface';

@Injectable()
export class InventoryService {
  private products: Product[] = [];

  async addProduct(addProductDto: AddProductDto): Promise<Product> {
    const newProduct: Product = {
      id: uuidv4(),
      name: addProductDto.name,
      description: addProductDto.description,
      price: addProductDto.price,
      quantity: addProductDto.quantity,
    };
    this.products.push(newProduct);
    return newProduct;
  }

  async getProducts(): Promise<Product[]> {
    return this.products;
  }

  async updateProductQuantity(id: string, updateProductQuantityDto: UpdateProductQuantityDto): Promise<Product> {
    const productIndex = this.products.findIndex((product) => product.id === id);
    if (productIndex === -1) {
      throw new Error('Product not found');
    }

    this.products[productIndex].quantity = updateProductQuantityDto.quantity;
    return this.products[productIndex];
  }
}
