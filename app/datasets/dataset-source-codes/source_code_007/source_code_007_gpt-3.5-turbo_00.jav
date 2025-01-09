public class PrimeChecker {
    public static boolean isPrime(int num) {
        if (num <= 1) {
            return false;
        }
        
        boolean[] prime = new boolean[num + 1];
        for (int i = 2; i <= num; i++) {
            prime[i] = true;
        }
        
        for (int p = 2; p * p <= num; p++) {
            if (prime[p]) {
                for (int i = p * p; i <= num; i += p) {
                    prime[i] = false;
                }
            }
        }
        
        return prime[num];
    }

    public static void main(String[] args) {
        int input1 = 7;
        int input2 = 12;

        System.out.println("Input: " + input1 + ", Output: " + isPrime(input1));
        System.out.println("Input: " + input2 + ", Output: " + isPrime(input2));
    }
}
