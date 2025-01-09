using System;
using System.Collections.Generic;

public class BracketChecker
{
    public static bool AreBracketsBalanced(string input)
    {
        // Create a stack to keep track of opening brackets
        Stack<char> bracketStack = new Stack<char>();

        // Dictionary to hold matching pairs of brackets
        Dictionary<char, char> bracketPairs = new Dictionary<char, char>()
        {
            { ')', '(' },
            { '}', '{' },
            { ']', '[' }
        };

        // Iterate over each character in the input string
        foreach (char ch in input)
        {
            // If the character is an opening bracket, push to stack
            if (ch == '(' || ch == '{' || ch == '[')
            {
                bracketStack.Push(ch);
            }
            else if (ch == ')' || ch == '}' || ch == ']') // If the character is a closing bracket
            {
                // If stack is empty or does not match with the corresponding opening bracket
                if (bracketStack.Count == 0 || bracketStack.Pop() != bracketPairs[ch])
                {
                    return false;
                }
            }
        }

        // If no unmatched brackets are left, return true
        return bracketStack.Count == 0;
    }

    public static void Main()
    {
        Console.WriteLine(AreBracketsBalanced("(a+[b*c]-{d/e})")); // Output: true
        Console.WriteLine(AreBracketsBalanced("(a+[b*c)-{d/e}]")); // Output: false
    }
}
