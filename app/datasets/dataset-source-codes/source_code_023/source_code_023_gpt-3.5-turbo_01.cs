using System;
using System.Collections.Generic;

class BracketChecker
{
    public static bool CheckBalancedBrackets(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return true;
        }

        Dictionary<char, char> bracketPairs = new Dictionary<char, char>
        {
            { '(', ')' },
            { '[', ']' },
            { '{', '}' }
        };

        Stack<char> stack = new Stack<char>();

        foreach (char c in input)
        {
            if (bracketPairs.ContainsValue(c))
            {
                if (stack.Count == 0 || bracketPairs[stack.Peek()] != c)
                {
                    return false;
                }
                stack.Pop();
            }
            else if (bracketPairs.ContainsKey(c))
            {
                stack.Push(c);
            }
        }

        return stack.Count == 0;
    }

    static void Main()
    {
        string input1 = "(a+[b*c]-{d/e})";
        string input2 = "(a+[b*c)-{d/e}]";

        Console.WriteLine($"Input: \"{input1}\"\nOutput: {CheckBalancedBrackets(input1)}\n");
        Console.WriteLine($"Input: \"{input2}\"\nOutput: {CheckBalancedBrackets(input2)}");
    }
}
