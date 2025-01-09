// NestJS module and required imports
import { Module, Injectable, Controller, Get, Post, Put, Body, Query, NotFoundException, HttpException, HttpStatus } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Repository } from 'typeorm';

// Product entity definition
@Entity()
class Product {
    @PrimaryGeneratedColumn()
    id: number;

    @Column({ unique: true })
    name: string;

    @Column()
    description: string;

    @Column()
    price: number;

    @Column()
    availableQuantity: number;
}

// DTOs for product operations
class CreateProductDto {
    name: string;
    description: string;
    price: number;
    availableQuantity: number;
}

class UpdateProductQuantityDto {
    id: number;
    newQuantity: number;
}

class ProductOrderabilityDto {
    id: number;
    quantity: number;
}

// Product service handling database operations
@Injectable()
class ProductService {
    constructor(@InjectRepository(Product) private productRepository: Repository<Product>) {}

    async addProduct(productData: CreateProductDto): Promise<Product> {
        return await this.productRepository.save(productData);
    }

    async updateProductQuantity(id: number, newQuantity: number): Promise<void> {
        const product = await this.productRepository.findOne(id);
        if (!product) throw new NotFoundException('Product not found');
        product.availableQuantity = newQuantity;
        await this.productRepository.save(product);
    }

    async listProducts(): Promise<Product[]> {
        return await this.productRepository.find();
    }

    async checkProductOrderability(id: number, quantity: number): Promise<boolean> {
        const product = await this.productRepository.findOne(id);
        if (!product) throw new NotFoundException('Product not found');
        return product.availableQuantity >= quantity;
    }
}

// Controller defining endpoints
@Controller('products')
class ProductController {
    constructor(private productService: ProductService) {}

    @Post()
    async addProduct(@Body() createProductDto: CreateProductDto) {
        try {
            const newProduct = await this.productService.addProduct(createProductDto);
            return { success: true, product: newProduct };
        } catch (error) {
            throw new HttpException('Failed to add product', HttpStatus.BAD_REQUEST);
        }
    }

    @Put()
    async updateProductQuantity(@Body() updateProductQuantityDto: UpdateProductQuantityDto) {
        try {
            await this.productService.updateProductQuantity(updateProductQuantityDto.id, updateProductQuantityDto.newQuantity);
            return { success: true };
        } catch (error) {
            throw new HttpException('Failed to update product quantity', HttpStatus.BAD_REQUEST);
        }
    }

    @Get()
    async getProducts() {
        return await this.productService.listProducts();
    }

    @Get('/check-orderable')
    async checkProductOrderability(@Query() query: ProductOrderabilityDto) {
        const isOrderable = await this.productService.checkProductOrderability(query.id, query.quantity);
        return { isOrderable };
    }
}

// Registering module and providers
@Module({
    imports: [TypeOrmModule.forFeature([Product])],
    controllers: [ProductController],
    providers: [ProductService],
})
export class ProductModule {}
