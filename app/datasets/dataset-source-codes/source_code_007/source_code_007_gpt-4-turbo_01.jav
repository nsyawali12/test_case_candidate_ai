import java.util.Scanner;

public class PrimeChecker {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int number = scanner.nextInt();
        scanner.close();

        if (number <= 1) {
            System.out.println("false");
            return;
        }

        boolean[] sieve = new boolean[number + 1];
        sievePrimes(sieve, number);

        System.out.println(!sieve[number]);
    }

    // Implements the Sieve of Eratosthenes algorithm
    private static void sievePrimes(boolean[] sieve, int limit) {
        // Initially assume all integers are prime
        for (int i = 2; i <= limit; i++) {
            sieve[i] = false;
        }

        // Mark non-primes with 'true' using Sieve of Eratosthenes
        for (int prime = 2; prime * prime <= limit; prime++) {
            if (!sieve[prime]) {
                for (int multiple = prime * prime; multiple <= limit; multiple += prime) {
                    sieve[multiple] = true;
                }
            }
        }
    }
}
