// inventory.module.ts
import { Module } from '@nestjs/common';
import { InventoryService } from './inventory.service';
import { InventoryController } from './inventory.controller';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Product } from './product.entity';
@Module({
  imports: [TypeOrmModule.forFeature([Product])],
  providers: [InventoryService],
  controllers: [InventoryController],
})
export class InventoryModule {}
// inventory.service.ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Product } from './product.entity';
import { CreateProductDto, UpdateProductDto } from './product.dto';

@Injectable()
export class InventoryService {
  constructor(
    @InjectRepository(Product)
    private readonly productRepository: Repository<Product>,
  ) {}

  async createProduct(createProductDto: CreateProductDto): Promise<Product> {
    const newProduct = new Product();
    newProduct.name = createProductDto.name;
    newProduct.description = createProductDto.description;
    newProduct.price = createProductDto.price;
    newProduct.availableQuantity = createProductDto.availableQuantity;

    return await this.productRepository.save(newProduct);
  }

  async updateProduct(
    productId: number,
    updateProductDto: UpdateProductDto,
  ): Promise<Product> {
    const product = await this.productRepository.findOne(productId);
    if (!product) {
      throw new NotFoundException('Product not found');
    }

    product.name = updateProductDto.name || product.name;
    product.description = updateProductDto.description || product.description;
    product.price = updateProductDto.price || product.price;
    product.availableQuantity =
      updateProductDto.availableQuantity || product.availableQuantity;

    return await this.productRepository.save(product);
  }

  async findAllProducts(): Promise<Product[]> {
    return await this.productRepository.find();
  }

  async getProductById(productId: number): Promise<Product> {
    const product = await this.productRepository.findOne(productId);
    if (!product) {
      throw new NotFoundException('Product not found');
    }
    return product;
  }

  async checkProductAvailability(productId: number, quantity: number): Promise<boolean> {
    const product = await this.productRepository.findOne(productId);
    if (!product) {
      throw new NotFoundException('Product not found');
    }
    return product.availableQuantity >= quantity;
  }
}