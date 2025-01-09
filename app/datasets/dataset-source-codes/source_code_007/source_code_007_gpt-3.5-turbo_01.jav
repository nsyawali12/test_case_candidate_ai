public class PrimeChecker {
    public boolean isPrime(int number) {
        if (number <= 1) {
            return false;
        }
        
        boolean[] isPrime = new boolean[number + 1];
        for (int i = 2; i <= number; i++) {
            isPrime[i] = true;
        }
        
        for (int i = 2; i * i <= number; i++) {
            if (isPrime[i]) {
                for (int j = i * i; j <= number; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        
        return isPrime[number];
    }
    
    public static void main(String[] args) {
        PrimeChecker primeChecker = new PrimeChecker();
        
        int input1 = 7;
        System.out.println("Input: " + input1 + ", Output: " + primeChecker.isPrime(input1));
        
        int input2 = 12;
        System.out.println("Input: " + input2 + ", Output: " + primeChecker.isPrime(input2));
    }
}
