// product.interface.ts
export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  quantity: number;
}

// create-product.dto.ts
export class CreateProductDto {
  readonly name: string;
  readonly description: string;
  readonly price: number;
  readonly quantity: number;
}

// update-product.dto.ts
export class UpdateProductDto {
  readonly quantity: number;
}

// inventory.service.ts
import { Injectable, NotFoundException } from '@nestjs/common';
import { Product } from './interfaces/product.interface';
import { CreateProductDto } from './dto/create-product.dto';
import { UpdateProductDto } from './dto/update-product.dto';

@Injectable()
export class InventoryService {
  private readonly products: Product[] = [];

  create(productDto: CreateProductDto): Product {
    const product: Product = {
      id: Date.now().toString(),
      ...productDto,
    };
    this.products.push(product);
    return product;
  }

  findAll(): Product[] {
    return this.products;
  }

  update(id: string, updateProductDto: UpdateProductDto): Product {
    const productIndex = this.products.findIndex(item => item.id === id);
    if (productIndex === -1) {
      throw new NotFoundException('Product not found');
    }
    this.products[productIndex].quantity = updateProductDto.quantity;
    return this.products[productIndex];
  }

  canBeOrdered(id: string, quantity: number): boolean {
    const productIndex = this.products.findIndex(item => item.id === id);
    if (productIndex === -1) {
      throw new NotFoundException('Product not found');
    }
    return this.products[productIndex].quantity >= quantity;
  }
}

// inventory.controller.ts
import { Body, Controller, Get, NotFoundException, Param, Post, Put } from '@nestjs/common';
import { CreateProductDto } from './dto/create-product.dto';
import { UpdateProductDto } from './dto/update-product.dto';
import { InventoryService } from './inventory.service';

@Controller('products')
export class InventoryController {
  constructor(private readonly inventoryService: InventoryService) { }

  @Post()
  create(@Body() createProductDto: CreateProductDto) {
    this.inventoryService.create(createProductDto);
  }

  @Get()
  findAll() {
    return this.inventoryService.findAll();
  }

  @Put(':id')
  update(@Param('id') id: string, @Body() updateProductDto: UpdateProductDto) {
    return this.inventoryService.update(id, updateProductDto);
  }

  @Get(':id/can-order/:quantity')
  canOrder(@Param('id') id: string, @Param('quantity') quantity: number) {
    const canBeOrdered = this.inventoryService.canBeOrdered(id, quantity);
    if (!canBeOrdered) {
      throw new NotFoundException('Not enough product in stock');
    }
    return { canBeOrdered };
  }
}

// inventory.module.ts
import { Module } from '@nestjs/common';
import { InventoryController } from './inventory.controller';
import { InventoryService } from './inventory.service';

@Module({
  controllers: [InventoryController],
  providers: [InventoryService],
})
export class InventoryModule { }
