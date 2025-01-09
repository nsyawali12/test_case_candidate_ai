import java.util.Arrays;

public class Main {
    public static void main(String[] args) {

        System.out.println(isPrime(7));  // true
        System.out.println(isPrime(12));  // false
    }

    public static boolean isPrime(int num) {

        // Use Sieve of Eratosthenes algorithm to find all primes up to the given num
        boolean[] primes = new boolean[num+1];
        Arrays.fill(primes, true);  // initialize all numbers as prime
        primes[0] = primes[1] = false;  // 0 and 1 are not primes 

        for (int p = 2; p * p <= num; p++){
            // If primes[p] is not changed, then it is a prime
            if (primes[p]){
                // Update all multiples of p
                for (int i = p * p; i <= num; i += p){
                    primes[i] = false;
                }
            }
        }
        // Return if the given num is prime or not
        return primes[num];
    }
}
