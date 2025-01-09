// Java program using Sieve of Eratosthenes algorithm
public class Main {
  
    static boolean SieveOfEratosthenes(int n, int[] primes) 
    { 
        // Create a boolean array "prime[0..n]" and initialize
        // all entries it as true. A value in prime[i] will
        // finally be false if i is Not a prime, else true.
        boolean[] prime = new boolean[n+1]; 
        for(int i=0;i<n;i++) 
            prime[i] = true; 
          
        for(int p = 2; p*p <=n; p++) 
        { 
            // If prime[p] is not changed, then it is a prime
            if(prime[p] == true) 
            { 
                // Update all multiples of p
                for(int i = p*p; i <= n; i += p) 
                    prime[i] = false; 
            } 
        } 
          
        // Check if number to be tested is prime
        return prime[primes[0]]; 
    } 
      
    // Driver Program 
    public static void main(String[] args) { 
        int n = 30; 
        int[] primes = new int[]{7, 12}; // Numbers to be tested
        for(int i=0;i<primes.length;i++){
            if (SieveOfEratosthenes(n, primes) == true)
                System.out.println(primes[i] +" "+ true);
            else
                System.out.println(primes[i] +" "+ false);
        }
    } 
}
