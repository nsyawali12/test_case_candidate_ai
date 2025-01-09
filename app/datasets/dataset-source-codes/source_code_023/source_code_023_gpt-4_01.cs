using System;
using System.Collections.Generic;

public class BalancedBrackets
{
    public bool IsBalanced(string input)
    {
        // Create a stack to hold opening brackets
        Stack<char> stack = new Stack<char>();

        for(int i = 0; i < input.Length; i++)
        {
            char c = input[i];

            // If the current char is an opening bracket, push it to the stack
            if(c == '(' || c == '{' || c == '[')
            {
                stack.Push(c);
            }
            // If the current char is a closing bracket
            else if(c == ')' || c == '}' || c == ']')
            {
                // If the stack is empty, the brackets are not balanced
                if(stack.Count == 0)
                {
                    return false;
                }

                // If the current closing bracket does not correspond to the last opening bracket, the brackets are not balanced
                if((c == ')' && stack.Peek() != '(') || (c == '}' && stack.Peek() != '{') || (c == ']' && stack.Peek() != '['))
                {
                    return false;
                }

                // Remove the last opening bracket from the stack
                stack.Pop();
            }
        }

        // If the stack is not empty, there is an opening bracket without a corresponding closing bracket
        return stack.Count == 0;
    }
}
