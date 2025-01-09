// inventory.module.ts
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Product } from './product.entity';
import { InventoryService } from './inventory.service';
import { InventoryController } from './inventory.controller';

@Module({
  imports: [TypeOrmModule.forFeature([Product])],
  controllers: [InventoryController],
  providers: [InventoryService],
})
export class InventoryModule {}

// inventory.service.ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Product } from './product.entity';
import { UpdateProductDto, CreateProductDto } from './dto';

@Injectable()
export class InventoryService {
  constructor(
    @InjectRepository(Product)
    private productRepository: Repository<Product>,
  ) {}

  async addProduct(createProductDto: CreateProductDto): Promise<Product> {
    const product = this.productRepository.create(createProductDto);
    return this.productRepository.save(product);
  }

  async updateProductQuantity(id: number, quantity: number): Promise<void> {
    await this.productRepository.update(id, { availableQuantity: quantity });
  }

  async findAllProducts(): Promise<Product[]> {
    return this.productRepository.find();
  }

  async checkProductAvailability(id: number, quantity: number): Promise<boolean> {
    const product = await this.productRepository.findOne(id);
    return product && product.availableQuantity >= quantity;
  }
}

// inventory.controller.ts
import { Controller, Get, Post, Put, Body, Param, Query } from '@nestjs/common';
import { InventoryService } from './inventory.service';
import { CreateProductDto, UpdateProductQuantityDto } from './dto';

@Controller('products')
export class InventoryController {
  constructor(private readonly inventoryService: InventoryService) {}

  @Post()
  async addProduct(@Body() createProductDto: CreateProductDto) {
    return this.inventoryService.addProduct(createProductDto);
  }

  @Get()
  async findAllProducts() {
    return this.inventoryService.findAllProducts();
  }

  @Put(':id/quantity')
  async updateProductQuantity(@Param('id') id: number, @Body() updateProductQuantityDto: UpdateProductQuantityDto) {
    await this.inventoryService.updateProductQuantity(id, updateProductQuantityDto.quantity);
  }

  @Get(':id/availability')
  async checkProductAvailability(@Param('id') id: number, @Query('quantity') quantity: number) {
    return { isAvailable: await this.inventoryService.checkProductAvailability(id, quantity) };
  }
}

// dto.ts
export class CreateProductDto {
  name: string;
  description: string;
  price: number;
  availableQuantity: number;
}

export class UpdateProductQuantityDto {
  quantity: number;
}

// product.entity.ts
import { Entity, Column, PrimaryGeneratedColumn } from 'typeorm';

@Entity()
export class Product {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @Column('text')
  description: string;

  @Column('float')
  price: number;

  @Column('int')
  availableQuantity: number;
}
