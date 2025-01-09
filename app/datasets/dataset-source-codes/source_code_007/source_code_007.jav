static boolean sieveOfEratosthenes(int checkNumber) {
  if (checkNumber == 0 || checkNumber == 1) return false;

  boolean prime[] = new boolean[checkNumber+1];
  for(int i = 1; i <= checkNumber; i++) 
    prime[i] = true;

  for(int j = 2; j*j <= checkNumber; j++){
    if(prime[j]) {
      for(int i = j*j; i <= checkNumber; i+=j) {
        prime[i] = false;
      }
    }
  }
  return prime[checkNumber];
}

public static void main(String[] args) {
  int checkNumber = 7;
  System.out.println(sieveOfEratosthenes(checkNumber));
}