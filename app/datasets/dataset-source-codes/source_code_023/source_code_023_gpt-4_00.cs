using System;
using System.Collections.Generic;

public class BracketsBalance
{
    // This function checks for balanced brackets in a string
    public static bool IsBalanced(string s)
    {
        // Using stack to store opening brackets
        Stack<char> bracketsStack = new Stack<char>();

        // Checking each character of string
        foreach(char ch in s)
        {
            switch(ch)
            {
                // Checking for opening brackets and pushing them to stack
                case '(':
                case '{':
                case '[':
                    bracketsStack.Push(ch);
                    break;

                // Checking for closing brackets
                case ')':
                    if (!bracketsStack.TryPop(out char opening) || opening != '(')
                    {
                        return false;
                    }
                    break;
                case '}':
                    if (!bracketsStack.TryPop(out opening) || opening != '{')
                    {
                        return false;
                    }
                    break;
                case ']':
                    if (!bracketsStack.TryPop(out opening) || opening != '[')
                    {
                        return false;
                    }
                    break;
            }
        }

        // If there are any opening brackets left in the stack, then expresssion is not balanced
        return bracketsStack.Count == 0;
    }
}
