using System;
using System.Collections.Generic;

public class Program
{
    public static bool IsBalancedBrackets(string input)
    {
        // Stack to keep track of opening brackets
        Stack<char> brackets = new Stack<char>();

        // Dictionary to hold the matching brackets
        Dictionary<char, char> matchingBrackets = new Dictionary<char, char>
        {
            { ')', '(' },
            { '}', '{' },
            { ']', '[' }
        };

        // Loop through each character in the string
        foreach (char ch in input)
        {
            if (ch == '(' || ch == '{' || ch == '[')
            {
                // If it's an opening bracket, push to stack
                brackets.Push(ch);
            }
            else if (ch == ')' || ch == '}' || ch == ']')
            {
                // If it's a closing bracket, check stack for the matching opening bracket
                if (brackets.Count == 0 || brackets.Pop() != matchingBrackets[ch])
                {
                    // If stack is empty or does not match, return false
                    return false;
                }
            }
        }

        // If the stack is empty, all brackets matched properly
        return brackets.Count == 0;
    }

    public static void Main()
    {
        Console.WriteLine(IsBalancedBrackets("(a+[b*c]-{d/e})"));  // Output: true
        Console.WriteLine(IsBalancedBrackets("(a+[b*c)-{d/e}]"));  // Output: false
    }
}
