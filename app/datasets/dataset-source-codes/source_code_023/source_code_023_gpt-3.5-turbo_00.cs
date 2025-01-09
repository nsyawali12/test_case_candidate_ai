using System;
using System.Collections.Generic;

class BracketChecker
{
    private readonly Dictionary<char, char> bracketPairs = new Dictionary<char, char>
    {
        { '(', ')' },
        { '[', ']' },
        { '{', '}' }
    };

    public bool CheckBalancedBrackets(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return true;
        }

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
}

class Program
{
    static void Main()
    {
        BracketChecker bracketChecker = new BracketChecker();

        string input1 = "(a+[b*c]-{d/e})";
        Console.WriteLine("Input: \"{0}\"", input1);
        Console.WriteLine("Output: {0}\n", bracketChecker.CheckBalancedBrackets(input1));

        string input2 = "(a+[b*c)-{d/e}]";
        Console.WriteLine("Input: \"{0}\"", input2);
        Console.WriteLine("Output: {0}", bracketChecker.CheckBalancedBrackets(input2));
    }
}
