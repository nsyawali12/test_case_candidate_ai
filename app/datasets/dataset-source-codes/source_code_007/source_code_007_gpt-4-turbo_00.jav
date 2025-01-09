import java.util.Arrays;

public class PrimeChecker {
    public static void main(String[] args) {
        int number1 = 7;
        int number2 = 12;

        System.out.println(isPrime(number1)); // Output: true
        System.out.println(isPrime(number2)); // Output: false
    }

    // Function to check if a number is prime using Sieve of Eratosthenes
    public static boolean isPrime(int n) {
        if (n <= 1) return false; // 0 and 1 are not prime numbers
        boolean[] isPrime = new boolean[n + 1];
        Arrays.fill(isPrime, true); // Assume all numbers are prime
        isPrime[0] = isPrime[1] = false; // 0 and 1 are not prime numbers

        for (int i = 2; i * i <= n; i++) {
            if (isPrime[i]) {
                for (int j = i * i; j <= n; j += i) {
                    isPrime[j] = false; // Marking the multiples of i as non-prime
                }
            }
        }

        return isPrime[n]; // Return the primality of n
    }
}
